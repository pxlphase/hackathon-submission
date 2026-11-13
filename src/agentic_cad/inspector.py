from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape

_FACE_TYPE_NAMES = {
    GeomAbs_Plane: "PLANE",
    GeomAbs_Cylinder: "CYLINDER",
    GeomAbs_Cone: "CONE",
    GeomAbs_Sphere: "SPHERE",
    GeomAbs_Torus: "TORUS",
    GeomAbs_BSplineSurface: "BSPLINE",
}


@dataclass
class CylinderFeature:
    kind: str  # "hole" or "boss"
    diameter: float
    axis: tuple[float, float, float]
    centre: tuple[float, float, float]  # Relative to bounding-box min corner
    length: float
    span_deg: float


@dataclass
class CadTelemetry:
    is_valid: bool
    solids: int
    shells: int
    faces: int
    edges: int
    vertices: int
    through_holes_genus: int
    bbox_min: tuple[float, float, float]
    bounding_box_mm: dict[str, float]
    volume_mm3: float
    surface_area_mm2: float
    face_types: dict[str, int]
    holes: list[CylinderFeature] = field(default_factory=list)
    bosses: list[CylinderFeature] = field(default_factory=list)


@dataclass
class FactsDiff:
    bbox_errs_pct: dict[str, float]
    sorted_bbox_max_err_pct: float
    max_bbox_err_pct: float
    axis_swapped_hint: bool
    vol_err_pct: float
    area_err_pct: float
    genus_diff: int
    missing_hole_groups: list[str]
    extra_hole_count: int
    missing_face_types: dict[str, int]
    missing_boss_groups: list[str] = field(default_factory=list)

    def summary_lines(self, telemetry: CadTelemetry, facts: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        t_box = facts.get("bounding_box_mm", {})
        c_box = telemetry.bounding_box_mm
        lines.append(
            f"- Validity: is_valid={telemetry.is_valid}, solids={telemetry.solids} (must be 1), "
            f"faces={telemetry.faces} (target={facts.get('faces', '?')})"
        )
        lines.append(
            f"- Bounding box mm: generated=({c_box['x']:.2f}, {c_box['y']:.2f}, {c_box['z']:.2f}) "
            f"vs target=({t_box.get('x', 0):.2f}, {t_box.get('y', 0):.2f}, {t_box.get('z', 0):.2f}) "
            f"[errors: dX={self.bbox_errs_pct['x']:+.1f}%, dY={self.bbox_errs_pct['y']:+.1f}%, dZ={self.bbox_errs_pct['z']:+.1f}%]"
        )
        if self.axis_swapped_hint:
            lines.append(
                "- ORIENTATION MISMATCH: Sorted dimensions match target much better than raw X/Y/Z — "
                "your part axes appear rotated/swapped relative to the canonical [0..X, 0..Y, 0..Z] view frame!"
            )
        t_vol = float(facts.get("volume_mm3", 0.0) or 0.0)
        t_area = float(facts.get("surface_area_mm2", 0.0) or 0.0)
        lines.append(
            f"- Volume mm^3: generated={telemetry.volume_mm3:.1f} vs target={t_vol:.1f} ({self.vol_err_pct:+.1f}%)"
        )
        lines.append(
            f"- Surface area mm^2: generated={telemetry.surface_area_mm2:.1f} vs target={t_area:.1f} ({self.area_err_pct:+.1f}%)"
        )
        t_genus = facts.get("through_holes_genus")
        if t_genus is not None:
            lines.append(
                f"- Through-holes genus (topology): generated={telemetry.through_holes_genus} vs target={t_genus} "
                f"(diff={self.genus_diff:+d})"
            )
        if self.missing_hole_groups:
            lines.append("- Missing/mismatched cylindrical holes vs facts.json:")
            for msg in self.missing_hole_groups[:8]:
                lines.append(f"  * {msg}")
        if self.missing_boss_groups:
            lines.append("- Missing/mismatched cylindrical shafts/bosses vs facts.json:")
            for msg in self.missing_boss_groups[:8]:
                lines.append(f"  * {msg}")
        if self.missing_face_types:
            ft_str = ", ".join(f"{k}: need {v}" for k, v in self.missing_face_types.items())
            lines.append(f"- Face-type gaps: {ft_str}")
        return lines


def load_step_shape(step_path: Path) -> TopoDS_Shape:
    """Read a STEP file into an OpenCASCADE TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def _count_unique(shape: TopoDS_Shape, shape_type: Any) -> int:
    shape_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, shape_type, shape_map)
    return int(shape_map.Extent())


def _canonical_axis(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    ax, ay, az = dx / norm, dy / norm, dz / norm
    # Canonical sign: first significant non-zero component positive
    for v in (ax, ay, az):
        if abs(v) > 1e-4:
            if v < 0:
                return (-ax, -ay, -az)
            break
    return (ax, ay, az)


def _merge_collinear_cylinders(items: list[CylinderFeature]) -> list[CylinderFeature]:
    """Merge split seam halves (e.g., two 180-deg faces belonging to one 360-deg bore)."""
    merged: list[CylinderFeature] = []
    for item in items:
        matched = False
        for existing in merged:
            if existing.kind != item.kind:
                continue
            if abs(existing.diameter - item.diameter) > max(0.08, 0.01 * existing.diameter):
                continue
            dot = abs(
                existing.axis[0] * item.axis[0]
                + existing.axis[1] * item.axis[1]
                + existing.axis[2] * item.axis[2]
            )
            if dot < 0.995:
                continue
            # Perpendicular distance between line and existing centre
            vx = item.centre[0] - existing.centre[0]
            vy = item.centre[1] - existing.centre[1]
            vz = item.centre[2] - existing.centre[2]
            proj = vx * existing.axis[0] + vy * existing.axis[1] + vz * existing.axis[2]
            px = vx - proj * existing.axis[0]
            py = vy - proj * existing.axis[1]
            pz = vz - proj * existing.axis[2]
            perp_dist = math.sqrt(px * px + py * py + pz * pz)
            if perp_dist <= 0.15 and existing.span_deg + item.span_deg <= 365.0:
                existing.span_deg = min(360.0, existing.span_deg + item.span_deg)
                existing.length = max(existing.length, item.length)
                matched = True
                break
        if not matched:
            merged.append(item)
    return merged


def inspect_shape(shape: TopoDS_Shape) -> CadTelemetry:
    """Extract ground-truth-compatible BRep telemetry from an OpenCASCADE TopoDS_Shape."""
    is_valid = bool(BRepCheck_Analyzer(shape).IsValid())
    solids = _count_unique(shape, TopAbs_SOLID)
    shells = _count_unique(shape, TopAbs_SHELL)
    faces = _count_unique(shape, TopAbs_FACE)
    edges = _count_unique(shape, TopAbs_EDGE)
    vertices = _count_unique(shape, TopAbs_VERTEX)

    # Bounding box via optimal geometry (no tessellation inflation)
    bnd = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bnd, False, False)
    if bnd.IsVoid():
        xmin = ymin = zmin = xmax = ymax = zmax = 0.0
    else:
        xmin, ymin, zmin, xmax, ymax, zmax = bnd.Get()

    vol_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol_props)
    volume_mm3 = float(vol_props.Mass())

    surf_props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf_props)
    surface_area_mm2 = float(surf_props.Mass())

    # Count degenerated edges (sphere/cone poles) so Euler characteristic matches true topology
    edge_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edge_map)
    deg_edges = 0
    for idx in range(1, edge_map.Extent() + 1):
        edge = TopoDS.Edge_s(edge_map.FindKey(idx))
        if BRep_Tool.Degenerated_s(edge):
            deg_edges += 1

    # Inspect unique faces for inner hole wires, surface types, and cylinder bores/shafts
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    extra_wires = 0
    face_types: dict[str, int] = {}
    raw_cyls: list[CylinderFeature] = []

    for idx in range(1, face_map.Extent() + 1):
        face = TopoDS.Face_s(face_map.FindKey(idx))
        exp_w = TopExp_Explorer(face, TopAbs_WIRE)
        w_cnt = 0
        while exp_w.More():
            w_cnt += 1
            exp_w.Next()
        if w_cnt > 1:
            extra_wires += w_cnt - 1

        adaptor = BRepAdaptor_Surface(face)
        stype = adaptor.GetType()
        type_name = _FACE_TYPE_NAMES.get(stype, "OTHER")
        face_types[type_name] = face_types.get(type_name, 0) + 1

        if stype == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            u1, u2 = adaptor.FirstUParameter(), adaptor.LastUParameter()
            v1, v2 = adaptor.FirstVParameter(), adaptor.LastVParameter()
            span_deg = math.degrees(abs(u2 - u1))
            if span_deg >= 175.0:
                u_mid = 0.5 * (u1 + u2)
                v_mid = 0.5 * (v1 + v2)
                pnt = gp_Pnt()
                normal = gp_Vec()
                BRepGProp_Face(face).Normal(u_mid, v_mid, pnt, normal)
                loc = cyl.Location()
                axis_dir = cyl.Axis().Direction()
                vx = pnt.X() - loc.X()
                vy = pnt.Y() - loc.Y()
                vz = pnt.Z() - loc.Z()
                proj = vx * axis_dir.X() + vy * axis_dir.Y() + vz * axis_dir.Z()
                rx = vx - proj * axis_dir.X()
                ry = vy - proj * axis_dir.Y()
                rz = vz - proj * axis_dir.Z()
                dot_out = rx * normal.X() + ry * normal.Y() + rz * normal.Z()
                kind = "boss" if dot_out > 0.0 else "hole"
                cx = loc.X() + v_mid * axis_dir.X() - xmin
                cy = loc.Y() + v_mid * axis_dir.Y() - ymin
                cz = loc.Z() + v_mid * axis_dir.Z() - zmin
                raw_cyls.append(
                    CylinderFeature(
                        kind=kind,
                        diameter=round(float(cyl.Radius()) * 2.0, 3),
                        axis=_canonical_axis(axis_dir.X(), axis_dir.Y(), axis_dir.Z()),
                        centre=(round(cx, 2), round(cy, 2), round(cz, 2)),
                        length=round(abs(v2 - v1), 2),
                        span_deg=round(span_deg, 1),
                    )
                )

    merged_cyls = _merge_collinear_cylinders(raw_cyls)
    holes = [c for c in merged_cyls if c.kind == "hole" and c.span_deg >= 350.0]
    bosses = [c for c in merged_cyls if c.kind == "boss" and c.span_deg >= 350.0]

    chi = vertices - (edges - deg_edges) + faces - extra_wires
    genus = max(0, (2 * shells - chi) // 2) if shells > 0 else 0

    return CadTelemetry(
        is_valid=is_valid,
        solids=solids,
        shells=shells,
        faces=faces,
        edges=edges,
        vertices=vertices,
        through_holes_genus=genus,
        bbox_min=(round(xmin, 4), round(ymin, 4), round(zmin, 4)),
        bounding_box_mm={
            "x": round(max(0.0, xmax - xmin), 3),
            "y": round(max(0.0, ymax - ymin), 3),
            "z": round(max(0.0, zmax - zmin), 3),
        },
        volume_mm3=round(volume_mm3, 2),
        surface_area_mm2=round(surface_area_mm2, 2),
        face_types=face_types,
        holes=holes,
        bosses=bosses,
    )


def inspect_step_file(step_path: Path) -> CadTelemetry:
    shape = load_step_shape(step_path)
    return inspect_shape(shape)


def _parse_axis(axis_val: Any) -> tuple[float, float, float] | None:
    if isinstance(axis_val, str):
        s = axis_val.strip().upper()
        if s == "X":
            return (1.0, 0.0, 0.0)
        if s == "Y":
            return (0.0, 1.0, 0.0)
        if s == "Z":
            return (0.0, 0.0, 1.0)
        if s.startswith("(") and s.endswith(")"):
            try:
                parts = [float(p.strip()) for p in s[1:-1].split(",")]
                if len(parts) == 3:
                    return _canonical_axis(parts[0], parts[1], parts[2])
            except ValueError:
                return None
    elif isinstance(axis_val, (list, tuple)) and len(axis_val) == 3:
        return _canonical_axis(float(axis_val[0]), float(axis_val[1]), float(axis_val[2]))
    return None


def _perp_and_3d_dist(
    centre_a: tuple[float, float, float],
    centre_b: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> tuple[float, float]:
    dx = centre_a[0] - centre_b[0]
    dy = centre_a[1] - centre_b[1]
    dz = centre_a[2] - centre_b[2]
    dist3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    proj = dx * axis[0] + dy * axis[1] + dz * axis[2]
    px = dx - proj * axis[0]
    py = dy - proj * axis[1]
    pz = dz - proj * axis[2]
    perp = math.sqrt(px * px + py * py + pz * pz)
    return perp, dist3d


def _match_cylinder_groups(
    candidates: list[CylinderFeature],
    groups: list[dict[str, Any]],
) -> tuple[list[str], set[int]]:
    missing_msgs: list[str] = []
    matched_cand_indices: set[int] = set()

    for grp in groups:
        tdia = float(grp.get("diameter", 0.0))
        tcnt = int(grp.get("count", 0))
        if tcnt <= 0:
            continue
        taxis = _parse_axis(grp.get("axis"))
        eligible = [
            idx
            for idx, ch in enumerate(candidates)
            if idx not in matched_cand_indices
            and abs(ch.diameter - tdia) <= max(0.3, 0.03 * tdia)
            and (taxis is None or abs(sum(a * b for a, b in zip(ch.axis, taxis))) >= 0.96)
        ]

        raw_centres = grp.get("centres", [])
        t_centres = [
            (float(c[0]), float(c[1]), float(c[2]))
            for c in raw_centres
            if isinstance(c, (list, tuple)) and len(c) == 3
        ]

        if not t_centres:
            found = min(tcnt, len(eligible))
            for idx in eligible[:found]:
                matched_cand_indices.add(idx)
            if found < tcnt:
                missing_msgs.append(
                    f"Expected {tcnt}x dia={tdia}mm along axis={grp.get('axis')} (found {found})"
                )
            continue

        pos_tol = max(2.0, 0.05 * tdia)
        pairs: list[tuple[float, float, int, int]] = []
        for c_idx in eligible:
            ch = candidates[c_idx]
            ref_axis = taxis or ch.axis
            for tc_idx, tc in enumerate(t_centres):
                perp_dist, dist3d = _perp_and_3d_dist(ch.centre, tc, ref_axis)
                pairs.append((perp_dist, dist3d, c_idx, tc_idx))
        pairs.sort()

        used_c: set[int] = set()
        used_tc: set[int] = set()
        for perp_dist, _dist3d, c_idx, tc_idx in pairs:
            if perp_dist > pos_tol:
                break
            if c_idx in used_c or tc_idx in used_tc:
                continue
            used_c.add(c_idx)
            used_tc.add(tc_idx)
            matched_cand_indices.add(c_idx)

        matched_listed = len(used_tc)
        unlisted_needed = max(0, tcnt - len(t_centres))
        remaining_eligible = [idx for idx in eligible if idx not in used_c]
        unlisted_matched = min(unlisted_needed, len(remaining_eligible))
        for idx in remaining_eligible[:unlisted_matched]:
            matched_cand_indices.add(idx)
            used_c.add(idx)

        found = matched_listed + unlisted_matched
        if matched_listed < min(tcnt, len(t_centres)) or found < tcnt:
            unmatched_details: list[str] = []
            for tc_idx, tc in enumerate(t_centres):
                if tc_idx in used_tc:
                    continue
                if eligible:
                    best_perp = min(
                        _perp_and_3d_dist(candidates[c_idx].centre, tc, taxis or candidates[c_idx].axis)[0]
                        for c_idx in eligible
                    )
                    unmatched_details.append(
                        f"{[round(tc[0], 1), round(tc[1], 1), round(tc[2], 1)]} (nearest radial_offset={best_perp:.1f}mm)"
                    )
                else:
                    unmatched_details.append(str([round(tc[0], 1), round(tc[1], 1), round(tc[2], 1)]))
                if len(unmatched_details) >= 3:
                    break
            if unmatched_details:
                missing_msgs.append(
                    f"Expected {tcnt}x dia={tdia}mm along axis={grp.get('axis')} (matched {found}/{tcnt}); "
                    f"missing/shifted centres={', '.join(unmatched_details)}"
                )
            else:
                missing_msgs.append(
                    f"Expected {tcnt}x dia={tdia}mm along axis={grp.get('axis')} (found {found}); "
                    f"sample centres={raw_centres[:3]}"
                )

    return missing_msgs, matched_cand_indices


def compare_against_facts(telemetry: CadTelemetry, facts: dict[str, Any] | None) -> FactsDiff:
    """Compute signed percentage & structural errors relative to facts.json."""
    if not facts:
        return FactsDiff(
            bbox_errs_pct={"x": 0.0, "y": 0.0, "z": 0.0},
            sorted_bbox_max_err_pct=0.0,
            max_bbox_err_pct=0.0,
            axis_swapped_hint=False,
            vol_err_pct=0.0,
            area_err_pct=0.0,
            genus_diff=0,
            missing_hole_groups=[],
            extra_hole_count=0,
            missing_face_types={},
            missing_boss_groups=[],
        )

    t_box = facts.get("bounding_box_mm", {})
    c_box = telemetry.bounding_box_mm
    bbox_errs: dict[str, float] = {}
    for ax in ("x", "y", "z"):
        tv = float(t_box.get(ax, 0.0) or 0.0)
        cv = float(c_box.get(ax, 0.0))
        bbox_errs[ax] = ((cv - tv) / tv * 100.0) if tv > 1e-6 else (100.0 if cv > 1e-3 else 0.0)

    max_bbox_err = max(abs(v) for v in bbox_errs.values())
    t_sorted = sorted(float(t_box.get(ax, 0.0) or 0.0) for ax in ("x", "y", "z"))
    c_sorted = sorted(float(c_box.get(ax, 0.0)) for ax in ("x", "y", "z"))
    sorted_errs = [
        abs(cv - tv) / tv * 100.0 if tv > 1e-6 else 0.0 for cv, tv in zip(c_sorted, t_sorted)
    ]
    sorted_max_err = max(sorted_errs) if sorted_errs else 0.0
    axis_swapped = max_bbox_err > 5.0 and sorted_max_err < 2.0

    t_vol = float(facts.get("volume_mm3", 0.0) or 0.0)
    vol_err_pct = ((telemetry.volume_mm3 - t_vol) / t_vol * 100.0) if t_vol > 1e-6 else 0.0

    t_area = float(facts.get("surface_area_mm2", 0.0) or 0.0)
    area_err_pct = ((telemetry.surface_area_mm2 - t_area) / t_area * 100.0) if t_area > 1e-6 else 0.0

    t_genus = facts.get("through_holes_genus")
    genus_diff = (telemetry.through_holes_genus - int(t_genus)) if t_genus is not None else 0

    missing_hole_groups, matched_hole_indices = _match_cylinder_groups(
        telemetry.holes,
        facts.get("cylindrical_holes_and_bores", []),
    )
    extra_holes = max(0, len(telemetry.holes) - len(matched_hole_indices))

    missing_boss_groups, _ = _match_cylinder_groups(
        telemetry.bosses,
        facts.get("cylindrical_shafts_and_bosses", []),
    )

    missing_face_types: dict[str, int] = {}
    t_ft = facts.get("face_types", {})
    for k, target_cnt in t_ft.items():
        if int(target_cnt) > 0 and telemetry.face_types.get(k, 0) == 0:
            missing_face_types[k] = int(target_cnt)

    return FactsDiff(
        bbox_errs_pct=bbox_errs,
        sorted_bbox_max_err_pct=round(sorted_max_err, 2),
        max_bbox_err_pct=round(max_bbox_err, 2),
        axis_swapped_hint=axis_swapped,
        vol_err_pct=round(vol_err_pct, 2),
        area_err_pct=round(area_err_pct, 2),
        genus_diff=genus_diff,
        missing_hole_groups=missing_hole_groups,
        extra_hole_count=extra_holes,
        missing_face_types=missing_face_types,
        missing_boss_groups=missing_boss_groups,
    )
