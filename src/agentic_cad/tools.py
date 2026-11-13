from __future__ import annotations

import io
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from agentic_cad.cadquery_ref import extract_python_code
from agentic_cad.executor import execute_cadquery_script
from agentic_cad.fallback import _extract_mask_loops
from agentic_cad.inspector import FactsDiff, CadTelemetry, compare_against_facts, inspect_shape, load_step_shape
from agentic_cad.renderer import (
    VIEW_CAMERAS,
    ViewEvalResult,
    _erode_1px,
    _extract_target_edge_mask,
    evaluate_views,
    tessellate_shape,
)
from agentic_cad.scorer import SurrogateScore, score_candidate

# Canonical camera axis mapping matching renderer.VIEW_CAMERAS:
# (h_axis, v_axis, h_flip, v_flip)
# h_flip=True: left image edge u=0 maps to h=H_len, right u=1 maps to h=0.
# v_flip=True: top image row maps to v=0, bottom row maps to v=V_len.
ORTHO_VIEW_SPECS: dict[str, tuple[str, str, bool, bool]] = {
    "view_0_front": ("x", "z", False, False),
    "view_1_back": ("x", "z", True, False),
    "view_2_left": ("y", "z", True, False),
    "view_3_right": ("y", "z", False, False),
    "view_4_top": ("x", "y", False, False),
    "view_5_bottom": ("x", "y", False, True),
}

CAD_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "measure_ortho_features",
            "description": (
                "Extract exact canonical [0..X, 0..Y, 0..Z] mm outer polygon corners, through-hole "
                "circles/slots, and internal crease pockets/bosses from one of the 6 orthographic views."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": list(ORTHO_VIEW_SPECS.keys()),
                        "description": "Orthographic view name to measure.",
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "description": "Use 'summary' for key contours/features or 'full' for denser vertices.",
                    },
                },
                "required": ["view"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_view_crop",
            "description": (
                "Crop a native-resolution 1024px view region with labeled millimeter coordinate axes. "
                "Use canonical [0..X, 0..Y, 0..Z] mm ranges for orthographic views or [0..1] for iso views."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": list(VIEW_CAMERAS.keys()),
                        "description": "Target view name.",
                    },
                    "h_min": {"type": "number", "description": "Horizontal min bound (mm on ortho, 0..1 on iso)."},
                    "v_min": {"type": "number", "description": "Vertical min bound (mm on ortho, 0..1 on iso)."},
                    "h_max": {"type": "number", "description": "Horizontal max bound (mm on ortho, 0..1 on iso)."},
                    "v_max": {"type": "number", "description": "Vertical max bound (mm on ortho, 0..1 on iso)."},
                },
                "required": ["view", "h_min", "v_min", "h_max", "v_max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cad_probe",
            "description": (
                "Execute a complete CadQuery 2.8 Python script (assigning a single solid to `result`) "
                "and return BRep validity, volume/area/bbox errors, missing holes/bosses, and 8-view Silhouette IoU."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete CadQuery Python code assigning the solid to `result`.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


@dataclass
class ProbeOutcome:
    ok: bool
    report: str
    code: str
    step_path: Path | None = None
    telemetry: CadTelemetry | None = None
    diff: FactsDiff | None = None
    view_eval: ViewEvalResult | None = None
    score: SurrogateScore | None = None


def _bbox_dims_mm(facts: dict[str, Any] | None) -> tuple[dict[str, float], bool]:
    if not facts:
        return {"x": 1.0, "y": 1.0, "z": 1.0}, False
    bbox = facts.get("bounding_box_mm") or {}
    try:
        x = float(bbox.get("x") or 0.0)
        y = float(bbox.get("y") or 0.0)
        z = float(bbox.get("z") or 0.0)
    except (TypeError, ValueError):
        return {"x": 1.0, "y": 1.0, "z": 1.0}, False
    if min(x, y, z) <= 0.1:
        return {"x": 1.0, "y": 1.0, "z": 1.0}, False
    return {"x": x, "y": y, "z": z}, True


def _load_crop_bounds(view_path: Path) -> tuple[Image.Image, np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    try:
        im = Image.open(view_path).convert("RGB")
    except Exception:
        return None
    rgb = np.asarray(im)
    fg = _erode_1px(np.any(rgb < 245, axis=-1))
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None
    return im, rgb, fg, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _fit_circle_2d(pts_mm: np.ndarray, min_pts: int = 8, px_step_mm: float = 0.5) -> dict[str, Any] | None:
    """Least-squares algebraic circle fit (centered for numerical stability)."""
    if len(pts_mm) < min_pts:
        return None
    x = pts_mm[:, 0].astype(np.float64)
    y = pts_mm[:, 1].astype(np.float64)
    span_x = float(x.max() - x.min())
    span_y = float(y.max() - y.min())
    if min(span_x, span_y) < 1e-4:
        return None

    mx = float(x.mean())
    my = float(y.mean())
    dx = x - mx
    dy = y - my
    rhs = dx * dx + dy * dy
    A = np.column_stack([dx, dy, np.ones_like(dx)])
    try:
        sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c + 0.25 * (a * a + b * b)
    if r_sq <= 1e-6:
        return None
    r = math.sqrt(r_sq)
    cx = mx + 0.5 * a
    cy = my + 0.5 * b
    if max(span_x, span_y) > 2.4 * r * 1.15:
        return None

    dists = np.hypot(x - cx, y - cy)
    errs = np.abs(dists - r)
    rms = float(np.sqrt(np.mean(errs * errs)))
    max_err = float(errs.max())

    angles = np.sort(np.arctan2(y - cy, x - cx))
    diffs = np.diff(angles)
    wrap_gap = float((angles[0] + 2.0 * math.pi) - angles[-1])
    max_gap = float(max(wrap_gap, float(diffs.max()) if len(diffs) else wrap_gap))
    coverage = max(0.0, 2.0 * math.pi - max_gap)

    rms_tol = max(0.35, 0.70 * px_step_mm, 0.032 * r)
    max_tol = max(1.0, 1.8 * px_step_mm, 0.095 * r)

    if rms <= rms_tol and max_err <= max_tol and coverage >= 0.72 * 2.0 * math.pi:
        return {
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "diameter_mm": round(2.0 * r, 2),
            "rms_mm": round(rms, 2),
            "coverage_deg": round(math.degrees(coverage), 0),
            "kind": "circle",
        }
    if rms <= rms_tol * 1.15 and max_err <= max_tol * 1.15 and coverage >= 0.42 * 2.0 * math.pi:
        return {
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "diameter_mm": round(2.0 * r, 2),
            "rms_mm": round(rms, 2),
            "coverage_deg": round(math.degrees(coverage), 0),
            "kind": "arc",
        }
    return None


def _downsample_bool_max(mask: np.ndarray, size: int) -> np.ndarray:
    """Preserve thin 1px strokes cleanly when downsampling a boolean mask."""
    h, w = mask.shape
    if max(h, w) > size:
        im = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(ImageFilter.MaxFilter(3))
    else:
        im = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    res = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(res) > 48


def _erode_n(mask: np.ndarray, steps: int = 2) -> np.ndarray:
    cur = mask
    h, w = cur.shape
    for _ in range(steps):
        p = np.pad(cur, 1, mode="constant", constant_values=False)
        cur = (
            p[1 : h + 1, 1 : w + 1]
            & p[0:h, 1 : w + 1]
            & p[2 : h + 2, 1 : w + 1]
            & p[1 : h + 1, 0:w]
            & p[1 : h + 1, 2 : w + 2]
            & p[0:h, 0:w]
            & p[0:h, 2 : w + 2]
            & p[2 : h + 2, 0:w]
            & p[2 : h + 2, 2 : w + 2]
        )
    return cur


def _extract_internal_crease_components(
    rgb_crop_aligned: np.ndarray,
    fg_crop_aligned: np.ndarray,
    h_len: float,
    v_len: float,
    known_hole_circles: list[dict[str, Any]],
    comp_res: int = 224,
) -> list[dict[str, Any]]:
    """Find internal BRep crease pockets, counterbores, and bosses inside silhouette."""
    crop_im = Image.fromarray(rgb_crop_aligned, mode="RGB")
    raw_edges = _extract_target_edge_mask(crop_im, rgb_crop_aligned)
    fg_small = np.asarray(
        Image.fromarray(fg_crop_aligned.astype(np.uint8) * 255, mode="L").resize(
            (comp_res, comp_res), Image.Resampling.NEAREST
        )
    ) > 127
    edge_small = _downsample_bool_max(raw_edges, comp_res)
    interior = _erode_n(fg_small, steps=3)
    internal = edge_small & interior
    ys, xs = np.where(internal)
    if len(xs) < 10:
        return []

    # Connect across 1px diagonal/bridge breaks via BFS within 2px radius
    visited = np.zeros((comp_res, comp_res), dtype=bool)
    min_span_px = max(6, int(0.035 * comp_res))
    comps: list[dict[str, Any]] = []
    denom = max(1, comp_res - 1)
    diag = math.hypot(h_len, v_len)
    px_step_mm = max(h_len, v_len) / denom

    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        q: deque[tuple[int, int]] = deque([(sy, sx)])
        visited[sy, sx] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            r_lo = max(0, r - 2)
            r_hi = min(comp_res - 1, r + 2)
            c_lo = max(0, c - 2)
            c_hi = min(comp_res - 1, c + 2)
            for nr in range(r_lo, r_hi + 1):
                for nc in range(c_lo, c_hi + 1):
                    if internal[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
        if len(pts) < 12:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        rs = arr[:, 0]
        cs = arr[:, 1]
        if max(float(rs.max() - rs.min()), float(cs.max() - cs.min())) < min_span_px:
            continue

        hs = (cs / denom) * h_len
        vs = (1.0 - rs / denom) * v_len
        pts_mm = np.column_stack([hs, vs])
        h_lo, h_hi = round(float(hs.min()), 2), round(float(hs.max()), 2)
        v_lo, v_hi = round(float(vs.min()), 2), round(float(vs.max()), 2)
        fit = _fit_circle_2d(pts_mm, min_pts=12, px_step_mm=px_step_mm)

        # Skip duplicates overlapping already reported through-hole boundaries
        if fit is not None:
            dup = False
            for kh in known_hole_circles:
                dist = math.hypot(fit["cx"] - kh["cx"], fit["cy"] - kh["cy"])
                if dist <= 0.04 * diag and abs(fit["diameter_mm"] - kh["diameter_mm"]) <= 0.12 * max(
                    fit["diameter_mm"], kh["diameter_mm"]
                ):
                    dup = True
                    break
            if dup:
                continue

        comps.append(
            {
                "h_range": [h_lo, h_hi],
                "v_range": [v_lo, v_hi],
                "span_score": max(h_hi - h_lo, v_hi - v_lo) + 0.25 * min(h_hi - h_lo, v_hi - v_lo),
                "circle": fit,
            }
        )

    comps.sort(key=lambda c: c["span_score"], reverse=True)
    return comps


def _dedup_close_vertices(pts: list[tuple[float, float]], min_dist_mm: float) -> list[tuple[float, float]]:
    if len(pts) <= 3:
        return pts
    out: list[tuple[float, float]] = [pts[0]]
    for pt in pts[1:]:
        if math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) >= min_dist_mm:
            out.append(pt)
    if len(out) > 3 and math.hypot(out[-1][0] - out[0][0], out[-1][1] - out[0][1]) < min_dist_mm:
        out.pop()
    return out


def _simplify_closed_to_max(pts: list[tuple[float, float]], max_verts: int, diag: float) -> list[tuple[float, float]]:
    """Simplify a closed polygon while preserving all full-perimeter corners (never truncating)."""
    from agentic_cad.fallback import _simplify_closed_loop

    cur = pts
    eps = 0.008 * diag
    while len(cur) > max_verts and eps < 0.08 * diag:
        cur = _simplify_closed_loop(cur, eps=eps)
        eps *= 1.35
    if len(cur) > max_verts:
        idxs = np.linspace(0, len(cur) - 1, max_verts, dtype=int).tolist()
        cur = [cur[i] for i in idxs]
    return cur


def analyze_ortho_view(
    view_path: Path,
    view_name: str,
    facts: dict[str, Any] | None,
    detail: str = "summary",
) -> dict[str, Any]:
    """Extract canonical 2D polygon vertices, through-holes, and internal creases in mm."""
    if view_name not in ORTHO_VIEW_SPECS:
        return {"error": f"Unknown orthographic view `{view_name}`. Valid: {sorted(ORTHO_VIEW_SPECS)}"}
    loaded = _load_crop_bounds(view_path)
    if loaded is None:
        return {"error": f"Could not read foreground silhouette from {view_path.name}."}
    _, rgb, fg, (x0, y0, x1, y1) = loaded
    h_ax, v_ax, h_flip, v_flip = ORTHO_VIEW_SPECS[view_name]
    dims, has_dims = _bbox_dims_mm(facts)
    h_len = dims[h_ax]
    v_len = dims[v_ax]
    unit = "mm" if has_dims else "norm"

    rgb_crop = rgb[y0 : y1 + 1, x0 : x1 + 1]
    fg_crop = fg[y0 : y1 + 1, x0 : x1 + 1]
    if h_flip:
        rgb_crop = rgb_crop[:, ::-1]
        fg_crop = fg_crop[:, ::-1]
    if v_flip:
        rgb_crop = rgb_crop[::-1, :]
        fg_crop = fg_crop[::-1, :]

    grid_res = 176 if detail == "full" else 144
    px_step_mm = max(h_len, v_len) / grid_res
    outers_uv, holes_uv = _extract_mask_loops(fg_crop, grid_res=grid_res)

    max_verts = 40 if detail == "full" else 18
    diag = math.hypot(h_len, v_len)
    outer_polys: list[list[list[float]]] = []
    for lp in outers_uv:
        mm_pts = [(round(u * h_len, 2), round(v * v_len, 2)) for u, v in lp]
        mm_pts = _dedup_close_vertices(mm_pts, min_dist_mm=0.008 * diag)
        mm_pts = _simplify_closed_to_max(mm_pts, max_verts=max_verts, diag=diag)
        mm_pts = [(round(x, 2), round(y, 2)) for x, y in mm_pts]
        outer_polys.append([[x, y] for x, y in mm_pts])

    hole_features: list[dict[str, Any]] = []
    hole_circles: list[dict[str, Any]] = []
    for lp in holes_uv:
        pts_arr = np.asarray([(u * h_len, v * v_len) for u, v in lp], dtype=np.float64)
        h_lo, h_hi = round(float(pts_arr[:, 0].min()), 2), round(float(pts_arr[:, 0].max()), 2)
        v_lo, v_hi = round(float(pts_arr[:, 1].min()), 2), round(float(pts_arr[:, 1].max()), 2)
        fit = _fit_circle_2d(pts_arr, min_pts=6, px_step_mm=px_step_mm)
        entry: dict[str, Any] = {"h_range": [h_lo, h_hi], "v_range": [v_lo, v_hi]}
        if fit is not None:
            entry["circle"] = fit
            hole_circles.append(fit)
        else:
            mm_pts = _dedup_close_vertices(
                [(round(float(p[0]), 2), round(float(p[1]), 2)) for p in pts_arr],
                min_dist_mm=0.012 * diag,
            )
            mm_pts = _simplify_closed_to_max(mm_pts, max_verts=12, diag=diag)
            entry["vertices"] = [[round(x, 2), round(y, 2)] for x, y in mm_pts]
        hole_features.append(entry)

    internal = _extract_internal_crease_components(rgb_crop, fg_crop, h_len, v_len, hole_circles)
    max_int = 14 if detail == "full" else 7
    return {
        "view": view_name,
        "h_axis": h_ax.upper(),
        "v_axis": v_ax.upper(),
        "h_len": round(h_len, 2),
        "v_len": round(v_len, 2),
        "unit": unit,
        "outer_polygons": outer_polys,
        "through_openings": hole_features,
        "internal_creases": internal[:max_int],
    }


def measure_ortho_features(
    views: list[Path],
    facts: dict[str, Any] | None,
    view_name: str,
    detail: str = "summary",
) -> str:
    """Format canonical mm contour vertices, through-holes, and internal creases for one ortho view."""
    vmap = {p.stem: p for p in views}
    vp = vmap.get(view_name)
    if vp is None:
        return json.dumps({"error": f"View `{view_name}` not found. Available: {sorted(vmap)}"})

    info = analyze_ortho_view(vp, view_name, facts, detail=detail)
    if "error" in info:
        return json.dumps(info)

    ha, va, u = info["h_axis"], info["v_axis"], info["unit"]
    lines = [
        f"Measured {view_name} in canonical frame ({ha}[0..{info['h_len']}{u}], {va}[0..{info['v_len']}{u}]):"
    ]
    for idx, poly in enumerate(info["outer_polygons"][:3]):
        pts_str = ", ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in poly)
        lines.append(f"- Outer contour #{idx + 1} ({len(poly)} verts [{ha},{va}]): [{pts_str}]")

    if info["through_openings"]:
        lines.append(f"- Through openings in silhouette ({len(info['through_openings'])}):")
        for h in info["through_openings"][:10]:
            if "circle" in h:
                c = h["circle"]
                lines.append(
                    f"  * {c['kind']} center=({ha}={c['cx']:.2f}, {va}={c['cy']:.2f}), "
                    f"dia={c['diameter_mm']:.2f}{u} (rms={c['rms_mm']:.2f}{u})"
                )
            else:
                lines.append(
                    f"  * non-circular opening {ha}[{h['h_range'][0]:.1f}..{h['h_range'][1]:.1f}] "
                    f"{va}[{h['v_range'][0]:.1f}..{h['v_range'][1]:.1f}], verts={h.get('vertices', [])}"
                )
    else:
        lines.append("- Through openings in silhouette: none")

    if info["internal_creases"]:
        lines.append(f"- Internal sharp creases/pockets/bosses ({len(info['internal_creases'])}):")
        for ic in info["internal_creases"]:
            fit = ic.get("circle")
            if fit is not None:
                lines.append(
                    f"  * internal {fit['kind']} ({fit['coverage_deg']:.0f}deg): "
                    f"center=({ha}={fit['cx']:.2f}, {va}={fit['cy']:.2f}), dia={fit['diameter_mm']:.2f}{u}, "
                    f"span {ha}[{ic['h_range'][0]:.1f}..{ic['h_range'][1]:.1f}] "
                    f"{va}[{ic['v_range'][0]:.1f}..{ic['v_range'][1]:.1f}]"
                )
            else:
                lines.append(
                    f"  * internal region {ha}[{ic['h_range'][0]:.1f}..{ic['h_range'][1]:.1f}] "
                    f"{va}[{ic['v_range'][0]:.1f}..{ic['v_range'][1]:.1f}]"
                )
    return "\n".join(lines)


def summarize_ortho_key_features(views: list[Path], facts: dict[str, Any] | None) -> str:
    """Compact Turn-0 key contour vertices + circular internal/through features across 3 primary planes."""
    _, has_dims = _bbox_dims_mm(facts)
    if not has_dims or not views:
        return ""
    vmap = {p.stem: p for p in views}
    planes = [
        ("Front XZ", "view_0_front", "view_1_back"),
        ("Right YZ", "view_3_right", "view_2_left"),
        ("Top XY", "view_4_top", "view_5_bottom"),
    ]
    lines: list[str] = []
    for label, primary, fallback_v in planes:
        vname = primary if primary in vmap else (fallback_v if fallback_v in vmap else "")
        if not vname:
            continue
        info = analyze_ortho_view(vmap[vname], vname, facts, detail="summary")
        if "error" in info:
            continue
        ha, va = info["h_axis"], info["v_axis"]
        parts: list[str] = []
        if info["outer_polygons"]:
            poly = info["outer_polygons"][0]
            if len(poly) > 4:
                shown = poly[:18]
                pts_str = ", ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in shown)
                suffix = ", ..." if len(poly) > 18 else ""
                parts.append(f"outer({len(poly)}v [{ha},{va}])=[{pts_str}{suffix}]")
        if info["through_openings"]:
            holes_desc = []
            for h in info["through_openings"][:3]:
                if "circle" in h:
                    c = h["circle"]
                    holes_desc.append(f"d={c['diameter_mm']:.1f}@({c['cx']:.1f},{c['cy']:.1f})")
                else:
                    holes_desc.append(
                        f"slot({ha}[{h['h_range'][0]:.1f}..{h['h_range'][1]:.1f}],"
                        f"{va}[{h['v_range'][0]:.1f}..{h['v_range'][1]:.1f}])"
                    )
            parts.append("openings=[" + "; ".join(holes_desc) + "]")
        int_circs = [ic for ic in info["internal_creases"] if ic.get("circle") is not None][:3]
        if int_circs:
            ic_desc = [
                f"{ic['circle']['kind']}(d={ic['circle']['diameter_mm']:.1f}@({ic['circle']['cx']:.1f},{ic['circle']['cy']:.1f}))"
                for ic in int_circs
            ]
            parts.append("internal=[" + "; ".join(ic_desc) + "]")
        if parts:
            lines.append(f"  * {label} ({vname}): " + " | ".join(parts))

    if not lines:
        return ""
    return (
        "### Orthographic Key Contours & Internal Creases (canonical `[0..X, 0..Y, 0..Z]` mm):\n"
        + "\n".join(lines)
    )


def zoom_view_crop(
    views: list[Path],
    facts: dict[str, Any] | None,
    view_name: str,
    h_min: float | None = None,
    v_min: float | None = None,
    h_max: float | None = None,
    v_max: float | None = None,
    out_px: int = 512,
) -> tuple[str, bytes | None]:
    """Crop high-res target view with millimeter coordinate axis ticks and gridlines."""
    vmap = {p.stem: p for p in views}
    vp = vmap.get(view_name)
    if vp is None:
        return json.dumps({"error": f"View `{view_name}` not found. Available: {sorted(vmap)}"}), None

    is_ortho = view_name in ORTHO_VIEW_SPECS
    dims, has_dims = _bbox_dims_mm(facts)

    try:
        im = Image.open(vp).convert("RGB")
    except Exception as exc:
        return json.dumps({"error": f"Failed to load `{view_name}`: {exc}"}), None
    rgb = np.asarray(im)
    img_h, img_w = rgb.shape[:2]

    if is_ortho:
        h_ax, v_ax, h_flip, v_flip = ORTHO_VIEW_SPECS[view_name]
        h_len = dims[h_ax]
        v_len = dims[v_ax]
        fg = _erode_1px(np.any(rgb < 245, axis=-1))
        ys, xs = np.where(fg)
        if len(xs) > 0:
            fx0, fy0, fx1, fy1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        else:
            fx0, fy0, fx1, fy1 = 0, 0, img_w - 1, img_h - 1

        req_h0 = 0.0 if h_min is None else float(h_min)
        req_v0 = 0.0 if v_min is None else float(v_min)
        req_h1 = h_len if h_max is None else float(h_max)
        req_v1 = v_len if v_max is None else float(v_max)
        if req_h0 > req_h1:
            req_h0, req_h1 = req_h1, req_h0
        if req_v0 > req_v1:
            req_v0, req_v1 = req_v1, req_v0

        # Pad requested box slightly so nearby boundary lines stay visible
        pad_h = 0.06 * max(req_h1 - req_h0, 0.05 * h_len)
        pad_v = 0.06 * max(req_v1 - req_v0, 0.05 * v_len)
        ch0 = max(0.0, req_h0 - pad_h)
        ch1 = min(h_len, req_h1 + pad_h)
        cv0 = max(0.0, req_v0 - pad_v)
        cv1 = min(v_len, req_v1 + pad_v)
        # Ensure non-degenerate window
        if (ch1 - ch0) < 0.03 * h_len:
            mid = 0.5 * (ch0 + ch1)
            ch0 = max(0.0, mid - 0.02 * h_len)
            ch1 = min(h_len, mid + 0.02 * h_len)
        if (cv1 - cv0) < 0.03 * v_len:
            mid = 0.5 * (cv0 + cv1)
            cv0 = max(0.0, mid - 0.02 * v_len)
            cv1 = min(v_len, mid + 0.02 * v_len)

        u0, u1 = ch0 / max(1e-6, h_len), ch1 / max(1e-6, h_len)
        w0, w1 = cv0 / max(1e-6, v_len), cv1 / max(1e-6, v_len)
        fw = max(1, fx1 - fx0)
        fh = max(1, fy1 - fy0)

        if h_flip:
            px0 = int(round(fx0 + (1.0 - u1) * fw))
            px1 = int(round(fx0 + (1.0 - u0) * fw))
            left_val, right_val = ch1, ch0
        else:
            px0 = int(round(fx0 + u0 * fw))
            px1 = int(round(fx0 + u1 * fw))
            left_val, right_val = ch0, ch1

        if v_flip:
            py0 = int(round(fy0 + w0 * fh))
            py1 = int(round(fy0 + w1 * fh))
            top_val, bot_val = cv0, cv1
        else:
            py0 = int(round(fy0 + (1.0 - w1) * fh))
            py1 = int(round(fy0 + (1.0 - w0) * fh))
            top_val, bot_val = cv1, cv0

        px0 = max(0, min(img_w - 2, px0))
        px1 = max(px0 + 2, min(img_w, px1 + 1))
        py0 = max(0, min(img_h - 2, py0))
        py1 = max(py0 + 2, min(img_h, py1 + 1))
        unit = "mm" if has_dims else "norm"
        axis_h_lbl, axis_v_lbl = h_ax.upper(), v_ax.upper()
    else:
        # Isometric view: normalized [0..1] coords (h left->right, v bottom->top)
        ch0 = max(0.0, 0.0 if h_min is None else float(h_min))
        ch1 = min(1.0, 1.0 if h_max is None else float(h_max))
        cv0 = max(0.0, 0.0 if v_min is None else float(v_min))
        cv1 = min(1.0, 1.0 if v_max is None else float(v_max))
        if ch0 > ch1:
            ch0, ch1 = ch1, ch0
        if cv0 > cv1:
            cv0, cv1 = cv1, cv0
        if (ch1 - ch0) < 0.04:
            mid = 0.5 * (ch0 + ch1)
            ch0, ch1 = max(0.0, mid - 0.03), min(1.0, mid + 0.03)
        if (cv1 - cv0) < 0.04:
            mid = 0.5 * (cv0 + cv1)
            cv0, cv1 = max(0.0, mid - 0.03), min(1.0, mid + 0.03)
        px0 = max(0, int(round(ch0 * (img_w - 1))))
        px1 = min(img_w, max(px0 + 2, int(round(ch1 * (img_w - 1))) + 1))
        py0 = max(0, int(round((1.0 - cv1) * (img_h - 1))))
        py1 = min(img_h, max(py0 + 2, int(round((1.0 - cv0) * (img_h - 1))) + 1))
        left_val, right_val = ch0, ch1
        top_val, bot_val = cv1, cv0
        unit = "norm"
        axis_h_lbl, axis_v_lbl = "U", "V"

    cropped = im.crop((px0, py0, px1, py1))
    header_h = 26
    margin_l = 54
    margin_b = 26
    margin_r = 12
    avail_w = out_px - margin_l - margin_r
    avail_h = out_px - header_h - margin_b

    cw, ch = cropped.size
    scale = min(avail_w / max(1, cw), avail_h / max(1, ch))
    nw = max(16, int(round(cw * scale)))
    nh = max(16, int(round(ch * scale)))
    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (out_px, out_px), (28, 32, 40))
    ox = margin_l + (avail_w - nw) // 2
    oy = header_h + (avail_h - nh) // 2
    canvas.paste(resized, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Title banner
    title = (
        f"ZOOM {view_name} | {axis_h_lbl}[{ch0:.1f}..{ch1:.1f}{unit}] "
        f"{axis_v_lbl}[{cv0:.1f}..{cv1:.1f}{unit}]"
    )
    draw.text((8, 6), title, fill=(242, 245, 250), font=font)

    # Coordinate gridlines and axis tick labels
    for i in range(5):
        frac = i / 4.0
        gx = int(round(ox + frac * (nw - 1)))
        gy = int(round(oy + frac * (nh - 1)))
        if 0 < i < 4:
            draw.line([(gx, oy), (gx, oy + nh - 1)], fill=(195, 200, 208), width=1)
            draw.line([(ox, gy), (ox + nw - 1, gy)], fill=(195, 200, 208), width=1)
        h_tick = left_val + frac * (right_val - left_val)
        v_tick = top_val + frac * (bot_val - top_val)
        draw.line([(gx, oy + nh), (gx, oy + nh + 4)], fill=(210, 215, 224), width=1)
        draw.text((max(margin_l - 8, gx - 12), oy + nh + 6), f"{h_tick:.1f}", fill=(220, 228, 240), font=font)
        draw.line([(ox - 4, gy), (ox, gy)], fill=(210, 215, 224), width=1)
        draw.text((4, max(header_h, gy - 6)), f"{axis_v_lbl}{v_tick:.1f}" if i == 0 else f"{v_tick:.1f}", fill=(220, 228, 240), font=font)

    draw.rectangle([ox - 1, oy - 1, ox + nw, oy + nh], outline=(110, 140, 180), width=1)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    legend = (
        f"Cropped {view_name}: {axis_h_lbl}=[{ch0:.2f}..{ch1:.2f}] {unit}, "
        f"{axis_v_lbl}=[{cv0:.2f}..{cv1:.2f}] {unit} ({cw}x{ch} native pixels -> {out_px}x{out_px} annotated PNG attached below)."
    )
    return legend, buf.getvalue()


def run_cad_probe(
    code: str,
    views: list[Path],
    facts: dict[str, Any] | None,
    work_dir: Path,
    probe_idx: int = 0,
    exec_timeout_s: float = 35.0,
    render_eval_px: int = 192,
) -> ProbeOutcome:
    """Execute a candidate CadQuery script safely and return structured BRep + visual telemetry."""
    clean_code = extract_python_code(code) if "```" in code else code.strip()
    step_path = work_dir / f"probe_{probe_idx}.stp"
    exec_res = execute_cadquery_script(
        clean_code,
        step_path,
        timeout_s=exec_timeout_s,
        allow_fillet_strip_fallback=False,
    )
    if not exec_res.ok or exec_res.step_path is None:
        tb_tail = "\n".join(exec_res.traceback_text.splitlines()[-10:])
        msg = f"cad_probe FAILED:\n{exec_res.error}"
        if tb_tail:
            msg += f"\nTraceback (tail):\n{tb_tail}"
        return ProbeOutcome(ok=False, report=msg, code=clean_code)

    try:
        shape = load_step_shape(exec_res.step_path)
        telemetry = inspect_shape(shape)
        diff = compare_against_facts(telemetry, facts)
        bbox = telemetry.bounding_box_mm
        diag = max(10.0, (bbox["x"] ** 2 + bbox["y"] ** 2 + bbox["z"] ** 2) ** 0.5)
        mesh = tessellate_shape(shape, diag_mm=diag)
        view_eval = evaluate_views(mesh, views, eval_size=min(256, render_eval_px))
        score = score_candidate(telemetry, facts, diff, view_eval)
    except Exception as exc:
        return ProbeOutcome(
            ok=False,
            report=f"cad_probe generated STEP but failed BRep inspection: {exc}",
            code=clean_code,
        )

    lines = [f"cad_probe SUCCEEDED | score={score.total:.3f} (valid={telemetry.is_valid}, solids={telemetry.solids})"]
    lines.extend(diff.summary_lines(telemetry, facts or {}))
    lines.append(
        f"- Visual 8-view Silhouette IoU: mean={view_eval.mean_silhouette_iou:.3f}, "
        f"min={view_eval.min_silhouette_iou:.3f}, edge_F1={view_eval.mean_edge_f1:.3f}"
    )
    if view_eval.worst_views:
        worst_str = ", ".join(
            f"{m.name} (IoU={m.silhouette_iou:.2f}, Edge={m.edge_f1:.2f})"
            for m in view_eval.worst_views[:3]
        )
        lines.append(f"- Worst matching views: {worst_str}")
    if view_eval.handedness_flip_detected:
        lines.append("- HANDEDNESS FLIP DETECTED: mirror error across multiple views!")

    return ProbeOutcome(
        ok=True,
        report="\n".join(lines),
        code=clean_code,
        step_path=exec_res.step_path,
        telemetry=telemetry,
        diff=diff,
        view_eval=view_eval,
        score=score,
    )


def parse_text_tool_calls(raw_text: str) -> tuple[str, list[dict[str, Any]]]:
    """Fallback parser for Qwen3.8 XML (<function=...><parameter=...>) or JSON <tool_call> tags."""
    calls: list[dict[str, Any]] = []
    if "<tool_call>" not in raw_text:
        return raw_text, calls

    cleaned = raw_text
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", raw_text, flags=re.DOTALL)
    for idx, block in enumerate(blocks):
        inner = block.strip()
        fn_match = re.search(r"<function=([^>\s]+)\s*>(.*?)</function>", inner, flags=re.DOTALL)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            body = fn_match.group(2)
            args: dict[str, Any] = {}
            for pm in re.finditer(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", body, flags=re.DOTALL):
                k = pm.group(1).strip()
                v_str = pm.group(2).strip()
                if k in {"h_min", "v_min", "h_max", "v_max"}:
                    try:
                        args[k] = float(v_str)
                        continue
                    except ValueError:
                        pass
                args[k] = v_str
            calls.append({"id": f"xml_call_{idx}", "name": fn_name, "arguments": args})
            continue
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                args_raw = obj.get("arguments") or {}
                if isinstance(args_raw, str):
                    args_raw = json.loads(args_raw)
                calls.append({"id": f"json_call_{idx}", "name": obj["name"], "arguments": args_raw})
        except Exception:
            continue

    if calls:
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", raw_text, flags=re.DOTALL).strip()
    return cleaned, calls
