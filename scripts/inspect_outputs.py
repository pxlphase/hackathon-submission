from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cadquery as cq

from agentic_cad.inspector import compare_against_facts, inspect_shape, load_step_shape
from agentic_cad.renderer import build_visual_diff_composite, evaluate_views, tessellate_shape
from agentic_cad.scorer import score_candidate

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect generated STEP solids against facts.json and 8 views.")
    parser.add_argument("--input_dir", type=Path, default=REPO_ROOT / "artifacts/inputs")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "artifacts/outputs")
    parser.add_argument("--render_dir", type=Path, default=None, help="Optional dir for all-view diff PNGs.")
    parser.add_argument("--stl_dir", type=Path, default=None, help="Optional dir to export .stl meshes for 3D viewers.")
    parser.add_argument("--eval_px", type=int, default=384)
    parser.add_argument("--cases", type=int, nargs="*", default=None, help="Optional case numbers, e.g. 1 4 5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dirs = sorted(
        (p for p in args.input_dir.glob("test_case_*") if (p / "description.md").is_file()),
        key=lambda p: int(p.name.removeprefix("test_case_")),
    )
    if args.cases:
        wanted = set(args.cases)
        case_dirs = [p for p in case_dirs if int(p.name.removeprefix("test_case_")) in wanted]

    if args.render_dir is not None:
        args.render_dir.mkdir(parents=True, exist_ok=True)
    if args.stl_dir is not None:
        args.stl_dir.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    ious: list[float] = []
    edges: list[float] = []

    for case_dir in case_dirs:
        name = case_dir.name
        stp_path = args.output_dir / f"{name}.stp"
        if not stp_path.is_file():
            print(f"{name}: MISSING ({stp_path})")
            scores.append(0.0)
            continue

        t0 = time.monotonic()
        facts_path = case_dir / "facts.json"
        facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.is_file() else {}
        views = sorted((case_dir / "images").glob("view_*.png"))

        shape = load_step_shape(stp_path)
        tel = inspect_shape(shape)
        diff = compare_against_facts(tel, facts)
        bbox = tel.bounding_box_mm
        diag = max(10.0, (bbox["x"] ** 2 + bbox["y"] ** 2 + bbox["z"] ** 2) ** 0.5)
        mesh = tessellate_shape(shape, diag_mm=diag)
        ve = evaluate_views(mesh, views, eval_size=args.eval_px)
        sc = score_candidate(tel, facts, diff, ve)

        scores.append(sc.total)
        ious.append(ve.mean_silhouette_iou)
        edges.append(ve.mean_edge_f1)

        t_genus = facts.get("through_holes_genus", "?")
        t_faces = facts.get("faces", "?")
        t_bb = facts.get("bounding_box_mm", {})
        size_kb = stp_path.stat().st_size / 1024.0
        dt = time.monotonic() - t0

        print(
            f"{name:12s} | score={sc.total:.3f} (vol={sc.volume_proxy:.2f} surf={sc.surface_proxy:.2f} "
            f"dim={sc.dimensions_score:.2f} hole={sc.through_holes_score:.2f}) | "
            f"iou={ve.mean_silhouette_iou:.3f} min_iou={ve.min_silhouette_iou:.3f} edge={ve.mean_edge_f1:.3f} | "
            f"vol_err={diff.vol_err_pct:+6.1f}% area_err={diff.area_err_pct:+6.1f}% "
            f"genus={tel.through_holes_genus}/{t_genus} faces={tel.faces}/{t_faces} "
            f"solids={tel.solids} valid={tel.is_valid} flip={ve.handedness_flip_detected} "
            f"size={size_kb:.0f}KB ({dt:.1f}s)"
        )
        print(
            f"             bbox=({bbox['x']:.2f}, {bbox['y']:.2f}, {bbox['z']:.2f}) "
            f"vs target=({float(t_bb.get('x', 0)):.2f}, {float(t_bb.get('y', 0)):.2f}, {float(t_bb.get('z', 0)):.2f}) | "
            f"face_types={tel.face_types}"
        )
        if diff.missing_hole_groups:
            print(f"             missing_holes: {'; '.join(diff.missing_hole_groups[:4])}")
        if diff.missing_boss_groups:
            print(f"             missing_bosses: {'; '.join(diff.missing_boss_groups[:4])}")
        if diff.missing_face_types:
            print(f"             missing_face_types: {diff.missing_face_types}")

        if args.render_dir is not None:
            png_bytes = build_visual_diff_composite(ve, worst_n=len(ve.metrics), tile_px=384)
            if png_bytes:
                (args.render_dir / f"{name}_diff.png").write_bytes(png_bytes)

        if args.stl_dir is not None:
            cq.exporters.export(cq.Shape.cast(shape), str(args.stl_dir / f"{name}.stl"), "STL")

    if scores:
        mean_sc = sum(scores) / len(scores)
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        mean_edge = sum(edges) / len(edges) if edges else 0.0
        print("-" * 100)
        print(f"MEAN ({len(scores)} cases): surrogate_score={mean_sc:.3f} | mean_iou={mean_iou:.3f} | mean_edge_f1={mean_edge:.3f}")


if __name__ == "__main__":
    main()
