from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
from PIL import Image

from agentic_cad.inspector import compare_against_facts, inspect_shape
from agentic_cad.renderer import evaluate_views, tessellate_shape
from agentic_cad.scorer import score_candidate


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


def _extract_mask_loops(
    mask_bool: np.ndarray,
    grid_res: int = 104,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
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
                best_idx = 0
                best_score = -999
                for idx, opt in enumerate(cand_list):
                    dx_out = opt[0] - curr[0]
                    dy_out = opt[1] - curr[1]
                    cross = dx_in * dy_out - dy_in * dx_out
                    dot = dx_in * dx_out + dy_in * dy_out
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
        simp = _simplify_closed_loop(cleaned, eps=0.0055)
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


def _load_fg_crop(path: Path, flip_horizontal: bool = False) -> np.ndarray | None:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    mask = _erode_1px(np.any(rgb < 245, axis=-1))
    if flip_horizontal:
        mask = mask[:, ::-1]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


def _fill_holes_fast(mask: np.ndarray, size: int = 96) -> np.ndarray:
    """Fill enclosed interior voids inside a mask crop via exterior BFS for circularity checks."""
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
        (size, size), Image.Resampling.NEAREST
    )
    arr = np.pad(np.asarray(im) > 127, 1, mode="constant", constant_values=False)
    H, W = arr.shape
    vis = np.zeros_like(arr, dtype=bool)
    q = deque([(0, 0)])
    vis[0, 0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc] and not arr[nr, nc]:
                vis[nr, nc] = True
                q.append((nr, nc))
    return ~vis[1:-1, 1:-1]


def _disk_iou(crop_filled: np.ndarray) -> float:
    h, w = crop_filled.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    r = 0.5 * min(h, w)
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
    inter = int(np.logical_and(crop_filled, disk).sum())
    union = int(np.logical_or(crop_filled, disk).sum())
    return float(inter / max(1, union))


def _build_revolve_candidate(
    view_map: dict[str, Path],
    X: float,
    Y: float,
    Z: float,
) -> cq.Shape | None:
    """Build a smooth revolved solid when two dimensions match and axial view is circular."""
    specs = [
        ("Z", X, Y, Z, "view_4_top", "view_5_bottom", [("view_0_front", False), ("view_3_right", False)]),
        ("Y", X, Z, Y, "view_0_front", "view_1_back", [("view_3_right", True), ("view_4_top", False)]),
        ("X", Y, Z, X, "view_3_right", "view_2_left", [("view_0_front", True), ("view_4_top", True)]),
    ]
    best_rev: cq.Shape | None = None
    best_iou = -1.0

    for ax, d1, d2, L, ax_v1, ax_v2, side_list in specs:
        if max(d1, d2) <= 1e-6 or abs(d1 - d2) / max(d1, d2) > 0.025:
            continue
        ax_path = view_map.get(ax_v1) or view_map.get(ax_v2)
        if ax_path is None:
            continue
        ax_crop = _load_fg_crop(ax_path)
        if ax_crop is None:
            continue
        ciou = _disk_iou(_fill_holes_fast(ax_crop, 96))
        if ciou < 0.85:
            continue

        R_max = 0.25 * (d1 + d2)
        N = 160
        prof_r: list[list[float]] = []
        for s_name, axial_is_col in side_list:
            sp = view_map.get(s_name)
            if sp is None:
                continue
            sc = _load_fg_crop(sp)
            if sc is None:
                continue
            H, W = sc.shape
            rs: list[float] = []
            if axial_is_col:
                for k in range(N):
                    c = min(W - 1, max(0, int(round(k / (N - 1) * (W - 1)))))
                    ys = np.where(sc[:, c])[0]
                    r = (int(ys.max()) - int(ys.min()) + 1) / max(1, H) * R_max if len(ys) else 0.0
                    rs.append(r)
            else:
                for k in range(N):
                    row_idx = min(H - 1, max(0, int(round((1.0 - k / (N - 1)) * (H - 1)))))
                    xs = np.where(sc[row_idx, :])[0]
                    r = (int(xs.max()) - int(xs.min()) + 1) / max(1, W) * R_max if len(xs) else 0.0
                    rs.append(r)
            prof_r.append(rs)

        if not prof_r:
            continue
        radii = [min(rs[k] for rs in prof_r) for k in range(N)]
        norm_pts = [(max(0.0, radii[k]) / max(1e-6, R_max), k / (N - 1)) for k in range(N)]
        simp_norm = _rdp(norm_pts, eps=0.004)
        prof_pts = [(0.0, 0.0)] + [(max(0.05, rn * R_max), zn * L) for rn, zn in simp_norm] + [(0.0, L)]
        clean_pts = [prof_pts[0]]
        for pt in prof_pts[1:]:
            if math.hypot(pt[0] - clean_pts[-1][0], pt[1] - clean_pts[-1][1]) > 0.02:
                clean_pts.append(pt)
        if clean_pts[-1][0] > 0.01:
            clean_pts.append((0.0, L))
        if len(clean_pts) < 3:
            continue

        if ax == "Z":
            plane = cq.Plane(origin=(0.5 * X, 0.5 * Y, 0.0), xDir=(1.0, 0.0, 0.0), normal=(0.0, -1.0, 0.0))
        elif ax == "Y":
            plane = cq.Plane(origin=(0.5 * X, 0.0, 0.5 * Z), xDir=(1.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        else:
            plane = cq.Plane(origin=(0.0, 0.5 * Y, 0.5 * Z), xDir=(0.0, 0.0, 1.0), normal=(0.0, 1.0, 0.0))

        try:
            solid = cq.Workplane(plane).polyline(clean_pts).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0)).val()
            if solid.isValid() and ciou > best_iou:
                best_iou = ciou
                best_rev = solid
        except Exception:
            continue
    return best_rev


def _extrude_view_prism(
    view_path: Path,
    u_len: float,
    v_len: float,
    depth: float,
    plane_name: str,
    flip_horizontal: bool = False,
    use_holes: bool = True,
    min_hole_norm_area: float = 0.004,
    grid_res: int = 104,
) -> tuple[cq.Shape | None, bool]:
    """Build a slightly oversized extruded BRep prism from a single orthographic view mask."""
    rgb = np.asarray(Image.open(view_path).convert("RGB"))
    mask = _erode_1px(np.any(rgb < 245, axis=-1))
    if flip_horizontal:
        mask = mask[:, ::-1]
    outers, holes = _extract_mask_loops(mask, grid_res=grid_res)
    sig_holes = [lp for lp in holes if abs(_signed_area(lp)) >= min_hole_norm_area]

    # Oversize along extrusion axis by +-0.25mm so orthogonal intersections avoid coplanar tangencies
    pad = 0.25
    ext_len = depth + 2.0 * pad

    def make_wp() -> cq.Workplane:
        if plane_name == "XY":
            return cq.Workplane(cq.Plane(origin=(0.0, 0.0, -pad), xDir=(1.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)))
        if plane_name == "XZ":
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
        return None, bool(sig_holes)
    prism = outer_solids[0]
    for extra in outer_solids[1:]:
        try:
            prism = prism.fuse(extra)
        except Exception:
            pass

    if use_holes:
        for hole_loop in sig_holes:
            pts = [(u * u_len, v * v_len) for (u, v) in hole_loop]
            try:
                cutter = make_wp().polyline(pts).close().extrude(ext_len).val()
                prism = prism.cut(cutter)
            except Exception:
                continue

    solids = prism.Solids()
    if not solids:
        return None, bool(sig_holes)
    return max(solids, key=lambda s: s.Volume()), bool(sig_holes)


def _parse_bore_group(
    grp: dict[str, Any],
    X: float,
    Y: float,
    Z: float,
    diag: float,
) -> tuple[bool, float, float, tuple[float, float, float], list[tuple[float, float, float]]] | None:
    dia = float(grp.get("diameter", 0.0) or 0.0)
    if dia <= 0.1:
        return None
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
        return None

    lengths = grp.get("lengths") or []
    avg_len = sum(float(l) for l in lengths) / len(lengths) if lengths else diag
    is_through = (
        (abs(dir_vec[0]) > 0.99 and avg_len >= 0.85 * X)
        or (abs(dir_vec[1]) > 0.99 and avg_len >= 0.85 * Y)
        or (abs(dir_vec[2]) > 0.99 and avg_len >= 0.85 * Z)
    )
    centres = [
        (float(ctr[0]), float(ctr[1]), float(ctr[2]))
        for ctr in grp.get("centres", [])
        if isinstance(ctr, (list, tuple)) and len(ctr) == 3
    ]
    if not centres:
        return None
    return is_through, dia, avg_len, dir_vec, centres


def _make_group_cylinders(
    dia: float,
    dir_vec: tuple[float, float, float],
    centres: list[tuple[float, float, float]],
    cyl_len: float,
) -> list[cq.Shape]:
    cyls: list[cq.Shape] = []
    for cx, cy, cz in centres:
        px = cx - 0.5 * cyl_len * dir_vec[0]
        py = cy - 0.5 * cyl_len * dir_vec[1]
        pz = cz - 0.5 * cyl_len * dir_vec[2]
        try:
            cyls.append(
                cq.Solid.makeCylinder(
                    dia * 0.5,
                    cyl_len,
                    cq.Vector(px, py, pz),
                    cq.Vector(*dir_vec),
                )
            )
        except Exception:
            continue
    return cyls


def _apply_group_cut(cur: cq.Shape, cyls: list[cq.Shape], min_vol: float) -> cq.Shape | None:
    if not cyls:
        return None
    try:
        comp = cq.Compound.makeCompound(cyls)
        res = cur.cut(comp)
        solids = res.Solids()
        if solids:
            candidate = max(solids, key=lambda s: s.Volume())
            if candidate.isValid() and candidate.Volume() > min_vol:
                return candidate
    except Exception:
        pass

    sub = cur
    for cyl in cyls:
        try:
            nxt = sub.cut(cyl)
            solids = nxt.Solids()
            if solids:
                best = max(solids, key=lambda s: s.Volume())
                if best.isValid() and best.Volume() > min_vol:
                    sub = best
        except Exception:
            continue
    return sub


def _cut_facts_bores(solid: cq.Shape, facts: dict[str, Any] | None, X: float, Y: float, Z: float) -> cq.Shape:
    """Apply facts.json cylindrical bore cuts safely, avoiding internal shell voids or spurious extra through-holes."""
    if not facts:
        return solid
    diag = math.sqrt(X * X + Y * Y + Z * Z) + 2.0
    tg = facts.get("through_holes_genus")
    tg_int = int(tg) if tg is not None else None

    groups = []
    for grp in facts.get("cylindrical_holes_and_bores", []):
        parsed = _parse_bore_group(grp, X, Y, Z, diag)
        if parsed is not None:
            groups.append(parsed)
    if not groups:
        return solid

    # Through-bores and largest bores first so primary topology establishes before small blind fastener holes
    groups.sort(key=lambda g: (g[0], g[1]), reverse=True)
    cur = solid
    cur_tel = inspect_shape(cur.wrapped)
    min_vol = max(1e-3, 0.02 * solid.Volume())

    for is_through, dia, avg_len, dir_vec, centres in groups:
        raw_len = diag if is_through else max(avg_len + 0.5, 1.0)
        cand = _apply_group_cut(cur, _make_group_cylinders(dia, dir_vec, centres, raw_len), min_vol)
        if cand is None:
            continue
        ct = inspect_shape(cand.wrapped)
        dg = ct.through_holes_genus - cur_tel.through_holes_genus

        # If a finite-length bore breaks single-shell continuity or shreds thin hull walls (dg > len(centres)),
        # try a clean punch length when it stays within target genus.
        if (not is_through) and (
            ct.shells != 1
            or (tg_int is not None and dg > len(centres) and cur_tel.through_holes_genus + len(centres) <= tg_int)
        ):
            ext_len = min(diag, max(avg_len * 2.2, avg_len + 0.25 * diag))
            cand_ext = _apply_group_cut(cur, _make_group_cylinders(dia, dir_vec, centres, ext_len), min_vol)
            if cand_ext is not None:
                ct_ext = inspect_shape(cand_ext.wrapped)
                dg_ext = ct_ext.through_holes_genus - cur_tel.through_holes_genus
                if (
                    ct_ext.is_valid
                    and ct_ext.solids == 1
                    and ct_ext.shells == 1
                    and (tg_int is None or ct_ext.through_holes_genus <= max(tg_int, cur_tel.through_holes_genus))
                ):
                    if ct.shells != 1 or dg_ext <= len(centres):
                        cand, ct = cand_ext, ct_ext

        if (
            ct.is_valid
            and ct.solids == 1
            and ct.shells == 1
            and (tg_int is None or ct.through_holes_genus <= max(tg_int, cur_tel.through_holes_genus))
        ):
            cur, cur_tel = cand, ct

    return cur


def _build_prism_hull(
    view_map: dict[str, Path],
    X: float,
    Y: float,
    Z: float,
    bounds_box: cq.Shape,
    use_holes: bool,
) -> tuple[cq.Shape, bool]:
    prisms: list[cq.Shape] = []
    any_sig_holes = False

    if "view_0_front" in view_map:
        p, hh = _extrude_view_prism(view_map["view_0_front"], X, Z, Y, "XZ", use_holes=use_holes)
        any_sig_holes = any_sig_holes or hh
        if p is not None:
            prisms.append(p)
    if "view_3_right" in view_map:
        p, hh = _extrude_view_prism(view_map["view_3_right"], Y, Z, X, "YZ", use_holes=use_holes)
        any_sig_holes = any_sig_holes or hh
        if p is not None:
            prisms.append(p)
    elif "view_2_left" in view_map:
        p, hh = _extrude_view_prism(
            view_map["view_2_left"], Y, Z, X, "YZ", flip_horizontal=True, use_holes=use_holes
        )
        any_sig_holes = any_sig_holes or hh
        if p is not None:
            prisms.append(p)
    if "view_4_top" in view_map:
        p, hh = _extrude_view_prism(view_map["view_4_top"], X, Y, Z, "XY", use_holes=use_holes)
        any_sig_holes = any_sig_holes or hh
        if p is not None:
            prisms.append(p)

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

    return best_solid, any_sig_holes


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
    diag = math.sqrt(X * X + Y * Y + Z * Z)

    view_map = {p.stem: p for p in views}
    bounds_box = cq.Solid.makeBox(X, Y, Z)
    candidates: list[cq.Shape] = []

    rev = _build_revolve_candidate(view_map, X, Y, Z)
    if rev is not None:
        candidates.append(_cut_facts_bores(rev, facts, X, Y, Z))

    tg = (facts or {}).get("through_holes_genus")
    if tg is None or int(tg) > 0:
        hull_holes, has_sig_holes = _build_prism_hull(view_map, X, Y, Z, bounds_box, use_holes=True)
        candidates.append(_cut_facts_bores(hull_holes, facts, X, Y, Z))
        if has_sig_holes:
            hull_plain, _ = _build_prism_hull(view_map, X, Y, Z, bounds_box, use_holes=False)
            candidates.append(_cut_facts_bores(hull_plain, facts, X, Y, Z))
    else:
        hull_plain, _ = _build_prism_hull(view_map, X, Y, Z, bounds_box, use_holes=False)
        candidates.append(_cut_facts_bores(hull_plain, facts, X, Y, Z))

    valid_cands: list[tuple[cq.Shape, Any]] = []
    for cand in candidates:
        try:
            tel = inspect_shape(cand.wrapped)
            if tel.is_valid and tel.solids == 1 and tel.shells == 1:
                valid_cands.append((cand, tel))
        except Exception:
            continue

    best_solid: cq.Shape
    if len(valid_cands) == 1:
        best_solid = valid_cands[0][0]
    elif len(valid_cands) > 1:
        scored: list[tuple[float, cq.Shape]] = []
        for cand, tel in valid_cands:
            try:
                diff = compare_against_facts(tel, facts)
                ve = evaluate_views(tessellate_shape(cand.wrapped, diag_mm=diag), views, eval_size=192)
                sc = score_candidate(tel, facts, diff, ve)
                scored.append((sc.total, cand))
            except Exception:
                continue
        best_solid = max(scored, key=lambda x: x[0])[1] if scored else valid_cands[0][0]
    else:
        try:
            fallback_box = _cut_facts_bores(bounds_box, facts, X, Y, Z)
            tel_b = inspect_shape(fallback_box.wrapped)
            best_solid = fallback_box if (tel_b.is_valid and tel_b.solids == 1) else bounds_box
        except Exception:
            best_solid = bounds_box

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(best_solid, str(output_path), "STEP")
    return output_path.is_file() and output_path.stat().st_size > 0
