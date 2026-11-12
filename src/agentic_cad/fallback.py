from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
from PIL import Image

from agentic_cad.inspector import inspect_shape


def _rdp(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(pts) <= 3:
        return pts
    ax, ay = pts[0]
    bx, by = pts[-1]
    dx, dy = bx - ax, by - ay
    seg_len = math.hypot(dx, dy)
    max_d = -1.0
    max_i = 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        if seg_len < 1e-9:
            d = math.hypot(px - ax, py - ay)
        else:
            d = abs(dy * px - dx * py + bx * ay - by * ax) / seg_len
        if d > max_d:
            max_d = d
            max_i = i
    if max_d > eps:
        left = _rdp(pts[: max_i + 1], eps)
        right = _rdp(pts[max_i:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _simplify_closed_loop(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(pts) <= 4:
        return pts
    # Split closed polygon at farthest vertex from index 0
    x0, y0 = pts[0]
    far_i = max(range(1, len(pts)), key=lambda i: (pts[i][0] - x0) ** 2 + (pts[i][1] - y0) ** 2)
    a = _rdp(pts[: far_i + 1], eps)
    b = _rdp(pts[far_i:] + [pts[0]], eps)
    merged = a[:-1] + b[:-1]
    return merged if len(merged) >= 3 else pts


def _signed_area(pts: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _extract_mask_loops(mask_bool: np.ndarray, grid_res: int = 84) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Extract normalized [0..1]x[0..1] (u left->right, v bottom->top) outer & hole polygon loops."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]], []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = mask_bool[y0 : y1 + 1, x0 : x1 + 1]

    # Resample tight bounding crop to compact binary grid
    im = Image.fromarray((crop.astype(np.uint8) * 255), mode="L").resize(
        (grid_res, grid_res), Image.Resampling.BILINEAR
    )
    arr = np.asarray(im) > 127
    # Pad with 1-cell white border cleanly
    grid = np.pad(arr, 1, mode="constant", constant_values=False)
    H, W = grid.shape

    # Collect directed boundary edges around foreground cells in (col, row_bottom_up)
    # Using standard pixel grid coordinates where y_img increases down -> convert to v = H - 1 - y_img
    # So CCW in (x, v) keeps foreground on the LEFT of directed segment.
    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add_seg(p0: tuple[int, int], p1: tuple[int, int]) -> None:
        outgoing.setdefault(p0, []).append(p1)

    for r in range(1, H - 1):
        v_bot = (H - 2) - (r - 1)  # pixel bottom edge in [0..grid_res]
        v_top = v_bot + 1
        for c in range(1, W - 1):
            if not grid[r, c]:
                continue
            u_left = c - 1
            u_right = c
            if not grid[r + 1, c]:  # neighbor below (smaller v) is empty -> bottom edge left->right
                add_seg((u_left, v_bot), (u_right, v_bot))
            if not grid[r, c + 1]:  # right neighbor empty -> right edge bottom->top
                add_seg((u_right, v_bot), (u_right, v_top))
            if not grid[r - 1, c]:  # top neighbor empty -> top edge right->left
                add_seg((u_right, v_top), (u_left, v_top))
            if not grid[r, c - 1]:  # left neighbor empty -> left edge top->bottom
                add_seg((u_left, v_top), (u_left, v_bot))

    loops: list[list[tuple[int, int]]] = []
    while outgoing:
        start = next(iter(outgoing))
        nxt_list = outgoing[start]
        nxt = nxt_list.pop()
        if not nxt_list:
            del outgoing[start]
        loop = [start]
        prev = start
        curr = nxt
        guard = 0
        while curr != start and guard < 20000:
            guard += 1
            loop.append(curr)
            cand_list = outgoing.get(curr)
            if not cand_list:
                break
            if len(cand_list) == 1:
                nxt_pt = cand_list.pop()
                del outgoing[curr]
            else:
                # Prefer turning right (smallest CW / non-crossing loop around checkerboard touch)
                dx_in = curr[0] - prev[0]
                dy_in = curr[1] - prev[1]
                # Order choices by turn angle
                best_idx = 0
                best_score = -999
                for idx, opt in enumerate(cand_list):
                    dx_out = opt[0] - curr[0]
                    dy_out = opt[1] - curr[1]
                    cross = dx_in * dy_out - dy_in * dx_out
                    dot = dx_in * dx_out + dy_in * dy_out
                    # Prefer right turn (cross < 0), then straight, then left turn
                    score = (-cross * 2) + dot
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                nxt_pt = cand_list.pop(best_idx)
                if not cand_list:
                    del outgoing[curr]
            prev = curr
            curr = nxt_pt
        if len(loop) >= 4:
            loops.append(loop)

    outers: list[list[tuple[float, float]]] = []
    holes: list[list[tuple[float, float]]] = []
    for raw_loop in loops:
        # Remove collinear mid-points first
        cleaned: list[tuple[float, float]] = []
        n = len(raw_loop)
        for i in range(n):
            p_prev = raw_loop[(i - 1) % n]
            p_cur = raw_loop[i]
            p_nxt = raw_loop[(i + 1) % n]
            if (p_cur[0] - p_prev[0]) * (p_nxt[1] - p_cur[1]) != (p_cur[1] - p_prev[1]) * (p_nxt[0] - p_cur[0]):
                cleaned.append((p_cur[0] / grid_res, p_cur[1] / grid_res))
        if len(cleaned) < 3:
            continue
        simp = _simplify_closed_loop(cleaned, eps=0.008)
        if len(simp) < 3:
            continue
        area = _signed_area(simp)
        if abs(area) < 0.001:
            continue
        if area > 0:
            outers.append(simp)
        else:
            holes.append(simp)

    if not outers:
        outers = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
    else:
        all_pts = [pt for lp in outers for pt in lp]
        u_lo = min(p[0] for p in all_pts)
        u_hi = max(p[0] for p in all_pts)
        v_lo = min(p[1] for p in all_pts)
        v_hi = max(p[1] for p in all_pts)
        du = u_hi - u_lo
        dv = v_hi - v_lo
        if du > 0.1 and dv > 0.1:
            outers = [[((u - u_lo) / du, (v - v_lo) / dv) for (u, v) in lp] for lp in outers]
            holes = [[((u - u_lo) / du, (v - v_lo) / dv) for (u, v) in lp] for lp in holes]
    return outers, holes


def _erode_1px(mask: np.ndarray) -> np.ndarray:
    """Trim 1-px outer dark outline stroke on high-res orthographic renders."""
    if min(mask.shape) < 256 or int(mask.sum()) < 200:
        return mask
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    h, w = mask.shape
    eroded = (
        p[1 : h + 1, 1 : w + 1]
        & p[0:h, 1 : w + 1]
        & p[2 : h + 2, 1 : w + 1]
        & p[1 : h + 1, 0:w]
        & p[1 : h + 1, 2 : w + 2]
    )
    return eroded if int(eroded.sum()) >= 50 else mask


def _extrude_view_prism(
    view_path: Path,
    u_len: float,
    v_len: float,
    depth: float,
    plane_name: str,
    flip_horizontal: bool = False,
) -> cq.Shape | None:
    """Build a slightly oversized extruded BRep prism from a single orthographic view mask."""
    rgb = np.asarray(Image.open(view_path).convert("RGB"))
    mask = _erode_1px(np.any(rgb < 245, axis=-1))
    if flip_horizontal:
        mask = mask[:, ::-1]
    outers, holes = _extract_mask_loops(mask, grid_res=84)

    # Oversize along extrusion axis by +-0.25mm so orthogonal intersections avoid coplanar tangencies
    pad = 0.25
    ext_len = depth + 2.0 * pad

    def make_wp() -> cq.Workplane:
        if plane_name == "XY":
            return cq.Workplane(cq.Plane(origin=(0.0, 0.0, -pad), xDir=(1.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)))
        if plane_name == "XZ":
            # Right-hand rule: xDir=(1,0,0) and yDir=(0,0,+1) requires normal=(0,-1,0); extrude from Y=depth+pad toward -Y
            return cq.Workplane(cq.Plane(origin=(0.0, depth + pad, 0.0), xDir=(1.0, 0.0, 0.0), normal=(0.0, -1.0, 0.0)))
        if plane_name == "YZ":
            return cq.Workplane(cq.Plane(origin=(-pad, 0.0, 0.0), xDir=(0.0, 1.0, 0.0), normal=(1.0, 0.0, 0.0)))
        raise ValueError(plane_name)

    outer_solids: list[cq.Shape] = []
    for loop in outers:
        pts = [(u * u_len, v * v_len) for (u, v) in loop]
        try:
            solid = make_wp().polyline(pts).close().extrude(ext_len).val()
            if solid.isValid():
                outer_solids.append(solid)
        except Exception:
            continue

    if not outer_solids:
        return None
    prism = outer_solids[0]
    for extra in outer_solids[1:]:
        try:
            prism = prism.fuse(extra)
        except Exception:
            pass

    for hole_loop in holes:
        pts = [(u * u_len, v * v_len) for (u, v) in hole_loop]
        try:
            cutter = make_wp().polyline(pts).close().extrude(ext_len).val()
            prism = prism.cut(cutter)
        except Exception:
            continue

    solids = prism.Solids()
    if not solids:
        return None
    return max(solids, key=lambda s: s.Volume())


def _cut_facts_bores(solid: cq.Shape, facts: dict[str, Any] | None, X: float, Y: float, Z: float) -> cq.Shape:
    if not facts:
        return solid
    diag = math.sqrt(X * X + Y * Y + Z * Z) + 2.0
    cutters: list[cq.Shape] = []
    for grp in facts.get("cylindrical_holes_and_bores", []):
        dia = float(grp.get("diameter", 0.0) or 0.0)
        if dia <= 0.1:
            continue
        ax = grp.get("axis")
        dir_vec: tuple[float, float, float] | None = None
        if isinstance(ax, str):
            s = ax.strip().upper()
            if s == "X":
                dir_vec = (1.0, 0.0, 0.0)
            elif s == "Y":
                dir_vec = (0.0, 1.0, 0.0)
            elif s == "Z":
                dir_vec = (0.0, 0.0, 1.0)
            elif s.startswith("(") and s.endswith(")"):
                try:
                    parts = [float(p.strip()) for p in s[1:-1].split(",")]
                    n = math.sqrt(sum(v * v for v in parts))
                    if len(parts) == 3 and n > 1e-6:
                        dir_vec = (parts[0] / n, parts[1] / n, parts[2] / n)
                except ValueError:
                    pass
        elif isinstance(ax, (list, tuple)) and len(ax) == 3:
            parts = [float(ax[0]), float(ax[1]), float(ax[2])]
            n = math.sqrt(sum(v * v for v in parts))
            if n > 1e-6:
                dir_vec = (parts[0] / n, parts[1] / n, parts[2] / n)
        if dir_vec is None:
            continue

        lengths = grp.get("lengths") or []
        avg_len = sum(float(l) for l in lengths) / len(lengths) if lengths else diag
        # Detect through-bores along bounding box axis vs blind pockets
        is_through = False
        if abs(dir_vec[0]) > 0.99 and avg_len >= 0.85 * X:
            is_through = True
        elif abs(dir_vec[1]) > 0.99 and avg_len >= 0.85 * Y:
            is_through = True
        elif abs(dir_vec[2]) > 0.99 and avg_len >= 0.85 * Z:
            is_through = True

        cyl_len = diag if is_through else max(avg_len + 0.5, 1.0)
        for ctr in grp.get("centres", []):
            if len(ctr) != 3:
                continue
            cx, cy, cz = float(ctr[0]), float(ctr[1]), float(ctr[2])
            px = cx - 0.5 * cyl_len * dir_vec[0]
            py = cy - 0.5 * cyl_len * dir_vec[1]
            pz = cz - 0.5 * cyl_len * dir_vec[2]
            try:
                cyl = cq.Solid.makeCylinder(
                    dia * 0.5,
                    cyl_len,
                    cq.Vector(px, py, pz),
                    cq.Vector(*dir_vec),
                )
                cutters.append(cyl)
            except Exception:
                continue

    if not cutters:
        return solid

    # Batch cut first; if batch fails, apply one by one safely
    try:
        comp = cq.Compound.makeCompound(cutters)
        res = solid.cut(comp)
        solids = res.Solids()
        if solids:
            candidate = max(solids, key=lambda s: s.Volume())
            if candidate.isValid() and candidate.Volume() > 1e-3:
                return candidate
    except Exception:
        pass

    cur = solid
    for cyl in cutters:
        try:
            nxt = cur.cut(cyl)
            solids = nxt.Solids()
            if solids:
                best = max(solids, key=lambda s: s.Volume())
                if best.isValid() and best.Volume() > 1e-3:
                    cur = best
        except Exception:
            continue
    return cur


def generate_fallback_step(
    views: list[Path],
    facts: dict[str, Any] | None,
    output_path: Path,
) -> bool:
    """Synthesize a deterministic single-solid STEP fallback purely on CPU (0 GPU VRAM)."""
    bbox = (facts or {}).get("bounding_box_mm", {})
    X = max(1.0, float(bbox.get("x", 50.0) or 50.0))
    Y = max(1.0, float(bbox.get("y", 50.0) or 50.0))
    Z = max(1.0, float(bbox.get("z", 50.0) or 50.0))

    view_map = {p.stem: p for p in views}
    prisms: list[cq.Shape] = []

    if "view_0_front" in view_map:
        p = _extrude_view_prism(view_map["view_0_front"], X, Z, Y, "XZ")
        if p is not None:
            prisms.append(p)
    if "view_3_right" in view_map:
        p = _extrude_view_prism(view_map["view_3_right"], Y, Z, X, "YZ")
        if p is not None:
            prisms.append(p)
    elif "view_2_left" in view_map:
        p = _extrude_view_prism(view_map["view_2_left"], Y, Z, X, "YZ", flip_horizontal=True)
        if p is not None:
            prisms.append(p)
    if "view_4_top" in view_map:
        p = _extrude_view_prism(view_map["view_4_top"], X, Y, Z, "XY")
        if p is not None:
            prisms.append(p)

    # Trim strictly to [0..X, 0..Y, 0..Z] box so oversized extrusions never expand dimensions
    bounds_box = cq.Solid.makeBox(X, Y, Z)
    best_solid: cq.Shape = bounds_box

    if prisms:
        cur: cq.Shape = bounds_box
        for pr in prisms:
            try:
                inter = cur.intersect(pr)
                solids = inter.Solids()
                if solids:
                    top_s = max(solids, key=lambda s: s.Volume())
                    if top_s.isValid() and top_s.Volume() > 0.02 * bounds_box.Volume():
                        cur = top_s
            except Exception:
                continue
        if cur.Volume() > 0.02 * bounds_box.Volume():
            best_solid = cur

    best_solid = _cut_facts_bores(best_solid, facts, X, Y, Z)

    try:
        tel = inspect_shape(best_solid.wrapped)
        if not tel.is_valid or tel.solids != 1:
            best_solid = _cut_facts_bores(bounds_box, facts, X, Y, Z)
    except Exception:
        best_solid = bounds_box

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(best_solid, str(output_path), "STEP")
    return output_path.is_file() and output_path.stat().st_size > 0
