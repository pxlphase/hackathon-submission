from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_cad.inspector import CadTelemetry, FactsDiff
from agentic_cad.renderer import ViewEvalResult


@dataclass
class SurrogateScore:
    total: float
    volume_proxy: float
    surface_proxy: float
    dimensions_score: float
    through_holes_score: float
    mean_silhouette_iou: float
    mean_edge_f1: float
    is_valid_single_solid: bool


def score_candidate(
    telemetry: CadTelemetry,
    facts: dict[str, Any] | None,
    diff: FactsDiff,
    view_eval: ViewEvalResult | None,
) -> SurrogateScore:
    """Estimate competition score in [0, 1] using Kickoff weights + 8-view silhouette proxy."""
    if not telemetry.is_valid or telemetry.solids != 1 or telemetry.volume_mm3 <= 1e-6:
        return SurrogateScore(
            total=0.0,
            volume_proxy=0.0,
            surface_proxy=0.0,
            dimensions_score=0.0,
            through_holes_score=0.0,
            mean_silhouette_iou=view_eval.mean_silhouette_iou if view_eval else 0.0,
            mean_edge_f1=view_eval.mean_edge_f1 if view_eval else 0.0,
            is_valid_single_solid=False,
        )

    mean_iou = view_eval.mean_silhouette_iou if view_eval else 0.5
    mean_edge = view_eval.mean_edge_f1 if view_eval else 0.5
    flip_penalty = 0.6 if (view_eval and view_eval.handedness_flip_detected) else 1.0

    if not facts:
        total = flip_penalty * (0.65 * mean_iou + 0.35 * mean_edge)
        return SurrogateScore(
            total=round(total, 4),
            volume_proxy=mean_iou,
            surface_proxy=mean_edge,
            dimensions_score=1.0,
            through_holes_score=1.0,
            mean_silhouette_iou=mean_iou,
            mean_edge_f1=mean_edge,
            is_valid_single_solid=True,
        )

    # Dimensions (0.20): 2% off loses 10%, 20% off loses all (rigid-alignment sorted dims)
    dim_score = max(0.0, 1.0 - diff.sorted_bbox_max_err_pct / 20.0)
    # Mild penalty if canonical axes are swapped relative to prompt view frame
    if diff.axis_swapped_hint:
        dim_score *= 0.85

    # Volume overlap proxy (0.30): combine mass agreement with 8-view silhouette IoU
    t_vol = float(facts.get("volume_mm3", 0.0) or 0.0)
    if t_vol > 1e-6:
        vol_match = max(0.0, 1.0 - abs(telemetry.volume_mm3 - t_vol) / max(telemetry.volume_mm3, t_vol))
    else:
        vol_match = 1.0
    vol_proxy = flip_penalty * (0.45 * vol_match + 0.55 * mean_iou)

    # Surface match proxy (0.30): combine surface area agreement with crease edge F1
    area_match = max(0.0, 1.0 - abs(diff.area_err_pct) / 25.0)
    surf_proxy = flip_penalty * (0.65 * area_match + 0.35 * mean_edge)

    # Through-holes / genus (0.20)
    t_genus = facts.get("through_holes_genus")
    if t_genus is None:
        holes_score = 1.0
    else:
        tg = int(t_genus)
        cg = telemetry.through_holes_genus
        if cg == tg:
            holes_score = 1.0
        elif tg == 0:
            holes_score = max(0.0, 1.0 - 0.5 * abs(cg - tg))
        else:
            holes_score = max(0.0, 1.0 - abs(cg - tg) / max(1, tg))

    total = 0.30 * vol_proxy + 0.30 * surf_proxy + 0.20 * dim_score + 0.20 * holes_score
    return SurrogateScore(
        total=round(total, 4),
        volume_proxy=round(vol_proxy, 4),
        surface_proxy=round(surf_proxy, 4),
        dimensions_score=round(dim_score, 4),
        through_holes_score=round(holes_score, 4),
        mean_silhouette_iou=round(mean_iou, 4),
        mean_edge_f1=round(mean_edge, 4),
        is_valid_single_solid=True,
    )
