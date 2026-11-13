from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
from PIL import Image

from agentic_cad.inspector import compare_against_facts, inspect_shape
from agentic_cad.renderer import VIEW_CAMERAS, evaluate_views, tessellate_shape
from agentic_cad.scorer import score_candidate

_MS_TABLE: dict[int, list[tuple[int, int]] | None] = {
    0: [],
    1: [(0, 3)],
    2: [(1, 0)],
    3: [(1, 3)],
    4: [(2, 1)],
    5: None,
    6: [(2, 0)],
    7: [(2, 3)],
    8: [(3, 2)],
    9: [(0, 2)],
    10: None,
    11: [(1, 2)],
    12: [(3, 1)],
    13: [(0, 1)],
    14: [(3, 0)],
    15: [],
}


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


def _cleanup_ring(
    pts: list[tuple[float, float]],
    min_dist: float = 1e-4,
    snap_bounds: bool = False,
    bound_tol: float = 0.0065,
) -> list[tuple[float, float]]:
    """Snap extreme outer bounds if requested and strip duplicate / collinear vertices."""
    work = list(pts)
    if snap_bounds:
        snapped: list[tuple[float, float]] = []
        for u, v in work:
            su = 0.0 if abs(u) <= bound_tol else (1.0 if abs(u - 1.0) <= bound_tol else u)
            sv = 0.0 if abs(v) <= bound_tol else (1.0 if abs(v - 1.0) <= bound_tol else v)
            snapped.append((su, sv))
        work = snapped

    dedup: list[tuple[float, float]] = []
    for pt in work:
        if not dedup or math.hypot(pt[0] - dedup[-1][0], pt[1] - dedup[-1][1]) >= min_dist:
            dedup.append(pt)
    if len(dedup) >= 3 and math.hypot(dedup[-1][0] - dedup[0][0], dedup[-1][1] - dedup[0][1]) < min_dist:
        dedup.pop()
    if len(dedup) < 3:
        return dedup

    out: list[tuple[float, float]] = []
    n = len(dedup)
    for i in range(n):
        p0 = out[-1] if out else dedup[(i - 1) % n]
        p1 = dedup[i]
        p2 = dedup[(i + 1) % n]
        l1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        l2 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        cross = abs((p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]))
        dot = (p1[0] - p0[0]) * (p2[0] - p1[0]) + (p1[1] - p0[1]) * (p2[1] - p1[1])
        if cross < 1e-4 * (l1 * l2) and dot > 0.0:
            continue
        out.append(p1)
    return out if len(out) >= 3 else dedup


def _extract_mask_loops(
    mask_bool: np.ndarray,
    grid_res: int = 160,
    eps: float = 0.0032,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Extract smooth subpixel [0..1]x[0..1] (u left->right, v bottom->top) outer & hole loops via Marching Squares."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]], []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = mask_bool[y0 : y1 + 1, x0 : x1 + 1]

    # Bilinear subpixel scalar field in [0, 1] padded by 1 empty cell so all boundary cycles close cleanly
    im = Image.fromarray((crop.astype(np.uint8) * 255), mode="L").resize(
        (grid_res, grid_res), Image.Resampling.BILINEAR
    )
    arr = np.asarray(im, dtype=np.float64) / 255.0
    field = np.pad(arr, 1, mode="constant", constant_values=0.0)
    iso = 0.5
    field = np.where(np.abs(field - iso) < 1e-4, iso + 1e-4, field)
    H, W = field.shape

    def interp(v0: float, v1: float) -> float:
        d = v1 - v0
        t = (iso - v0) / d if abs(d) > 1e-9 else 0.5
        return max(0.01, min(0.99, t))

    def edge_pt_and_key(r: int, c: int, edge_idx: int) -> tuple[tuple[str, int, int], tuple[float, float]]:
        v_bot = (H - 1) - (r + 1)
        v_top = (H - 1) - r
        if edge_idx == 0:
            t = interp(float(field[r + 1, c]), float(field[r + 1, c + 1]))
            return ("h", r + 1, c), (c + t, float(v_bot))
        if edge_idx == 1:
            t = interp(float(field[r + 1, c + 1]), float(field[r, c + 1]))
            return ("v", r, c + 1), (float(c + 1), v_bot + t)
        if edge_idx == 2:
            t = interp(float(field[r, c]), float(field[r, c + 1]))
            return ("h", r, c), (c + t, float(v_top))
        t = interp(float(field[r + 1, c]), float(field[r, c]))
        return ("v", r, c), (float(c), v_bot + t)

    nxt_edge: dict[tuple[str, int, int], tuple[str, int, int]] = {}
    edge_coords: dict[tuple[str, int, int], tuple[float, float]] = {}

    for r in range(H - 1):
        for c in range(W - 1):
            f_nw = float(field[r, c])
            f_ne = float(field[r, c + 1])
            f_se = float(field[r + 1, c + 1])
            f_sw = float(field[r + 1, c])
            idx = (
                (8 if f_nw > iso else 0)
                | (4 if f_ne > iso else 0)
                | (2 if f_se > iso else 0)
                | (1 if f_sw > iso else 0)
            )
            if idx == 0 or idx == 15:
                continue
            segs = _MS_TABLE[idx]
            if segs is None:
                c_in = 0.25 * (f_nw + f_ne + f_se + f_sw) > iso
                if idx == 5:
                    segs = [(0, 1), (2, 3)] if c_in else [(0, 3), (2, 1)]
                else:
                    segs = [(3, 0), (1, 2)] if c_in else [(1, 0), (3, 2)]
            for e_in, e_out in segs:
                k_in, pt_in = edge_pt_and_key(r, c, e_in)
                k_out, pt_out = edge_pt_and_key(r, c, e_out)
                edge_coords[k_in] = pt_in
                edge_coords[k_out] = pt_out
                nxt_edge[k_in] = k_out

    loops: list[list[tuple[float, float]]] = []
    while nxt_edge:
        start = next(iter(nxt_edge))
        curr = start
        loop: list[tuple[float, float]] = []
        guard = 0
        while curr in nxt_edge and guard < 40000:
            guard += 1
            nxt = nxt_edge.pop(curr)
            loop.append(edge_coords[curr])
            curr = nxt
            if curr == start:
                break
        if len(loop) >= 3:
            loops.append(loop)

    outers: list[list[tuple[float, float]]] = []
    holes: list[list[tuple[float, float]]] = []
    denom = float(max(1, grid_res))
    for raw_loop in loops:
        norm = [(pt[0] / denom, pt[1] / denom) for pt in raw_loop]
        simp = _simplify_closed_loop(norm, eps=eps)
        if len(simp) < 3:
            continue
        area = _signed_area(simp)
        if abs(area) < 0.0008:
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
            outers = [
                _cleanup_ring(
                    [((u - u_lo) / du, (v - v_lo) / dv) for (u, v) in lp],
                    min_dist=1e-4,
                    snap_bounds=True,
                )
                for lp in outers
            ]
            holes = [
                _cleanup_ring(
                    [((u - u_lo) / du, (v - v_lo) / dv) for (u, v) in lp],
                    min_dist=1e-4,
                    snap_bounds=False,
                )
                for lp in holes
            ]
            outers = [lp for lp in outers if len(lp) >= 3 and abs(_signed_area(lp)) >= 0.0008]
            holes = [lp for lp in holes if len(lp) >= 3 and abs(_signed_area(lp)) >= 0.0005]
            if not outers:
                outers = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
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


def _fit_circle_closed_pts(pts_mm: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Fit a full closed circular loop (cx, cy, r) in mm if all points tightly follow a single circle."""
    if len(pts_mm) < 8:
        return None
    arr = np.asarray(pts_mm, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    span_x = float(x.max() - x.min())
    span_y = float(y.max() - y.min())
    if min(span_x, span_y) < 0.5 or abs(span_x - span_y) / max(span_x, span_y) > 0.16:
        return None

    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    rhs = dx * dx + dy * dy
    A = np.column_stack([dx, dy, np.ones_like(dx)])
    try:
        sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c + 0.25 * (a * a + b * b)
    if r_sq <= 1e-4:
        return None
    r = math.sqrt(r_sq)
    if r < 0.25 or max(span_x, span_y) > 2.35 * r:
        return None
    cx, cy = mx + 0.5 * a, my + 0.5 * b
    errs = np.abs(np.hypot(x - cx, y - cy) - r)
    rms = float(np.sqrt(np.mean(errs * errs)))
    max_err = float(errs.max())

    angles = np.sort(np.arctan2(y - cy, x - cx))
    wrap_gap = float((angles[0] + 2.0 * math.pi) - angles[-1])
    diffs = np.diff(angles)
    max_gap = float(max(wrap_gap, float(diffs.max()) if len(diffs) else wrap_gap))
    coverage = max(0.0, 2.0 * math.pi - max_gap)
    if (
        rms <= max(0.24, 0.018 * r)
        and max_err <= max(0.65, 0.048 * r)
        and coverage >= 0.78 * 2.0 * math.pi
    ):
        return cx, cy, r
    return None


def _fit_circle_arc_run(
    run: list[tuple[float, float]],
    diag: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
    """Fit 1 or 2 monotonic circular threePointArc pairs (mid_pt, end_pt) to a smooth open point run."""
    if len(run) < 4:
        return None
    arr = np.asarray(run, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    rhs = dx * dx + dy * dy
    A = np.column_stack([dx, dy, np.ones_like(dx)])
    try:
        sol, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c + 0.25 * (a * a + b * b)
    if r_sq <= 1e-4:
        return None
    r = math.sqrt(r_sq)
    cx, cy = mx + 0.5 * a, my + 0.5 * b
    chord = math.hypot(run[-1][0] - run[0][0], run[-1][1] - run[0][1])
    if r < 0.35 or r > 50.0 * max(diag, chord, 1.0):
        return None

    errs = np.abs(np.hypot(x - cx, y - cy) - r)
    rms = float(np.sqrt(np.mean(errs * errs)))
    max_err = float(errs.max())
    if rms > max(0.18, 0.012 * r, 0.003 * diag) or max_err > max(0.42, 0.028 * r, 0.006 * diag):
        return None

    # Unwrap vertex angles and ensure monotonic progression around (cx, cy)
    raw_ang = np.arctan2(y - cy, x - cx)
    deltas = np.diff(raw_ang)
    deltas = (deltas + math.pi) % (2.0 * math.pi) - math.pi
    if not (np.all(deltas > 1e-4) or np.all(deltas < -1e-4)):
        return None
    total_sweep = float(deltas.sum())
    if abs(total_sweep) < math.radians(10.0) or abs(total_sweep) > math.radians(330.0):
        return None

    t0 = float(raw_ang[0])
    n_arcs = 2 if abs(total_sweep) > math.radians(140.0) else 1
    arcs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for k in range(n_arcs):
        a0 = t0 + total_sweep * (k / n_arcs)
        a1 = t0 + total_sweep * ((k + 1) / n_arcs)
        am = 0.5 * (a0 + a1)
        mid_pt = (cx + r * math.cos(am), cy + r * math.sin(am))
        end_pt = run[-1] if k == n_arcs - 1 else (cx + r * math.cos(a1), cy + r * math.sin(a1))
        arcs.append((mid_pt, end_pt))
    return arcs


def _extrude_loop_solid(
    make_wp: Any,
    pts_mm: list[tuple[float, float]],
    ext_len: float,
    is_hole: bool = False,
    axis_facts_bores: list[tuple[float, float, float]] | None = None,
) -> cq.Shape | None:
    """Extrude a 2D closed contour using exact circles, threePointArcs, B-splines, or safe polyline fallback."""
    if len(pts_mm) < 3:
        return None
    xs = [p[0] for p in pts_mm]
    ys = [p[1] for p in pts_mm]
    diag = max(1e-3, math.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    pts = _cleanup_ring(pts_mm, min_dist=max(0.03, 0.0005 * diag), snap_bounds=False)
    if len(pts) < 3:
        return None

    # 1. Exact analytical full-circle extrusion for round holes or cylindrical outer silhouettes
    circ = _fit_circle_closed_pts(pts)
    if circ is not None:
        cx, cy, r = circ
        if is_hole and axis_facts_bores:
            for fcx, fcy, fdia in axis_facts_bores:
                if (
                    math.hypot(fcx - cx, fcy - cy) <= max(1.2, 0.10 * fdia)
                    and abs(fdia - 2.0 * r) <= max(0.65, 0.045 * fdia)
                ):
                    cx, cy, r = fcx, fcy, 0.5 * fdia
                    break
        try:
            solid = make_wp().center(cx, cy).circle(r).extrude(ext_len).val()
            if solid.isValid() and len(solid.Solids()) == 1 and solid.Volume() > 1e-6:
                return solid
        except Exception:
            pass

    # 2. Smooth piecewise wire: split at sharp corners and long-to-short tangent transitions
    n = len(pts)
    edge_lens = [
        math.hypot(pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1])
        for i in range(n)
    ]
    turns_deg: list[float] = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        vx1, vy1 = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
        vx2, vy2 = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
        l1 = max(1e-9, math.hypot(vx1, vy1))
        l2 = max(1e-9, math.hypot(vx2, vy2))
        cos_t = max(-1.0, min(1.0, (vx1 * vx2 + vy1 * vy2) / (l1 * l2)))
        turns_deg.append(math.degrees(math.acos(cos_t)))

    long_thresh = max(3.5, 0.10 * diag)
    corner_flags: list[bool] = []
    for i in range(n):
        l_in = edge_lens[(i - 1) % n]
        l_out = edge_lens[i]
        t_deg = turns_deg[i]
        is_c = t_deg >= 26.0
        if not is_c and t_deg >= 2.5 and max(l_in, l_out) >= long_thresh and min(l_in, l_out) < 0.45 * max(l_in, l_out):
            is_c = True
        corner_flags.append(is_c)

    if any(corner_flags):
        # Rotate ring so index 0 starts cleanly on the sharpest corner (never mid-curve)
        best_rot = max(
            (i for i, fl in enumerate(corner_flags) if fl),
            key=lambda i: (turns_deg[i], max(edge_lens[(i - 1) % n], edge_lens[i])),
        )
        ring = pts[best_rot:] + pts[:best_rot]
        c_rot = corner_flags[best_rot:] + corner_flags[:best_rot]
        breaks = [i for i in range(1, n) if c_rot[i]] + [n]

        try:
            wp = make_wp().moveTo(ring[0][0], ring[0][1])
            start_idx = 0
            for idx_b, end_idx in enumerate(breaks):
                is_last_seg = idx_b == (len(breaks) - 1)
                run = [ring[k % n] for k in range(start_idx, end_idx + 1)]
                run_len = len(run)
                if run_len <= 2:
                    if not is_last_seg:
                        wp = wp.lineTo(run[-1][0], run[-1][1])
                else:
                    interior_turn = sum(turns_deg[(best_rot + k) % n] for k in range(start_idx + 1, end_idx))
                    arcs = _fit_circle_arc_run(run, diag=diag) if (run_len >= 4 and interior_turn >= 10.0) else None
                    if arcs is not None:
                        for mid_pt, end_pt in arcs:
                            wp = wp.threePointArc(mid_pt, end_pt)
                    elif interior_turn >= 10.0 and run_len >= 4:
                        wp = wp.spline(run[1:], includeCurrent=True)
                    else:
                        if not is_last_seg:
                            wp = wp.lineTo(run[-1][0], run[-1][1])
                start_idx = end_idx

            solid = wp.close().extrude(ext_len).val()
            if solid.isValid() and len(solid.Solids()) == 1 and solid.Volume() > 1e-6:
                return solid
        except Exception:
            pass

    # 3. Clean polygonal fallback
    try:
        solid = make_wp().polyline(pts).close().extrude(ext_len).val()
        if solid.isValid() and len(solid.Solids()) == 1 and solid.Volume() > 1e-6:
            return solid
    except Exception:
        pass
    return None


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


def _axis_facts_bore_hints(facts: dict[str, Any] | None, plane_name: str) -> list[tuple[float, float, float]]:
    """Return canonical 2D (h, v, dia) coordinates of facts bores aligned with a given ortho plane."""
    if not facts:
        return []
    expected_ax = {"XY": "Z", "XZ": "Y", "YZ": "X"}[plane_name]
    h_idx, v_idx = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[plane_name]
    hints: list[tuple[float, float, float]] = []
    for grp in facts.get("cylindrical_holes_and_bores", []):
        ax = str(grp.get("axis", "")).strip().upper()
        if ax != expected_ax:
            continue
        dia = float(grp.get("diameter", 0.0) or 0.0)
        if dia <= 0.1:
            continue
        for c in grp.get("centres", []):
            if isinstance(c, (list, tuple)) and len(c) == 3:
                hints.append((float(c[h_idx]), float(c[v_idx]), dia))
    return hints


def _extrude_view_prism(
    view_path: Path,
    u_len: float,
    v_len: float,
    depth: float,
    plane_name: str,
    flip_horizontal: bool = False,
    use_holes: bool = True,
    min_hole_norm_area: float = 0.0018,
    grid_res: int = 160,
    facts: dict[str, Any] | None = None,
    skip_facts_cyl_holes: bool = False,
) -> tuple[cq.Shape | None, bool]:
    """Build a slightly oversized smooth extruded BRep prism from a single orthographic view mask."""
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

    bore_hints = _axis_facts_bore_hints(facts, plane_name)
    outer_solids: list[cq.Shape] = []
    for loop in outers:
        pts = [(u * u_len, v * v_len) for (u, v) in loop]
        solid = _extrude_loop_solid(make_wp, pts, ext_len, is_hole=False)
        if solid is not None and solid.isValid():
            outer_solids.append(solid)

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
            if skip_facts_cyl_holes and bore_hints:
                circ = _fit_circle_closed_pts(pts)
                if circ is not None:
                    cx, cy, r = circ
                    if any(
                        math.hypot(fcx - cx, fcy - cy) <= max(2.0, 0.15 * fdia)
                        and abs(fdia - 2.0 * r) <= max(1.2, 0.12 * fdia)
                        for fcx, fcy, fdia in bore_hints
                    ):
                        continue
            cutter = _extrude_loop_solid(
                make_wp,
                pts,
                ext_len,
                is_hole=True,
                axis_facts_bores=bore_hints,
            )
            if cutter is None:
                continue
            try:
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


def _is_major_through_bore(
    is_through: bool,
    dia: float,
    avg_len: float,
    dir_vec: tuple[float, float, float],
    X: float,
    Y: float,
    Z: float,
) -> bool:
    if is_through:
        return True
    min_dim = min(X, Y, Z)
    max_axis = max(abs(dir_vec[0]), abs(dir_vec[1]), abs(dir_vec[2]))
    if max_axis < 0.98 and avg_len >= 0.45 * dia and dia >= 0.06 * min_dim:
        return True
    if abs(dir_vec[0]) > 0.99 and avg_len >= 0.70 * X and dia >= 0.12 * min_dim:
        return True
    if abs(dir_vec[1]) > 0.99 and avg_len >= 0.70 * Y and dia >= 0.12 * min_dim:
        return True
    if abs(dir_vec[2]) > 0.99 and avg_len >= 0.70 * Z and dia >= 0.12 * min_dim:
        return True
    return False


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

    # Prioritize primary / large through-bores before small blind fastener holes so topology cannot be starved
    groups.sort(
        key=lambda g: (
            _is_major_through_bore(g[0], g[1], g[2], g[3], X, Y, Z),
            g[1] * g[2],
            g[1],
        ),
        reverse=True,
    )
    cur = solid
    cur_tel = inspect_shape(cur.wrapped)
    min_vol = max(1e-3, 0.02 * solid.Volume())

    for is_through, dia, avg_len, dir_vec, centres in groups:
        is_major = _is_major_through_bore(is_through, dia, avg_len, dir_vec, X, Y, Z)
        raw_len = diag if is_through else max(avg_len + 0.5, 1.0)
        cand = _apply_group_cut(cur, _make_group_cylinders(dia, dir_vec, centres, raw_len), min_vol)
        if cand is None:
            continue
        ct = inspect_shape(cand.wrapped)
        dg = ct.through_holes_genus - cur_tel.through_holes_genus

        # If a major bore stops right at an oblique/curved surface (dg == 0 while genus < target)
        # or fractures shell continuity cleanly inside genus budget, try a controlled punch length.
        need_major_punch = (
            (not is_through)
            and is_major
            and (
                ct.shells != 1
                or (
                    tg_int is not None
                    and dg == 0
                    and cur_tel.through_holes_genus + len(centres) <= tg_int
                )
                or (
                    tg_int is not None
                    and dg > len(centres)
                    and cur_tel.through_holes_genus + len(centres) <= tg_int
                )
            )
        )
        if need_major_punch:
            for ext_len in (
                min(diag, max(avg_len * 1.45, avg_len + 25.0)),
                min(diag, max(avg_len * 2.2, avg_len + 0.25 * diag)),
            ):
                cand_ext = _apply_group_cut(cur, _make_group_cylinders(dia, dir_vec, centres, ext_len), min_vol)
                if cand_ext is None:
                    continue
                ct_ext = inspect_shape(cand_ext.wrapped)
                dg_ext = ct_ext.through_holes_genus - cur_tel.through_holes_genus
                accept_ext = (
                    (ct.shells != 1 and 0 <= dg_ext <= len(centres))
                    or (dg == 0 and 0 < dg_ext <= len(centres))
                    or (dg > len(centres) and dg_ext <= len(centres))
                )
                if (
                    ct_ext.is_valid
                    and ct_ext.solids == 1
                    and ct_ext.shells == 1
                    and (tg_int is None or ct_ext.through_holes_genus <= max(tg_int, cur_tel.through_holes_genus))
                    and accept_ext
                ):
                    cand, ct = cand_ext, ct_ext
                    break

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
    facts: dict[str, Any] | None = None,
    skip_facts_cyl_holes: bool = False,
) -> tuple[cq.Shape, bool]:
    prisms: list[cq.Shape] = []
    any_sig_holes = False

    if "view_0_front" in view_map:
        p, hh = _extrude_view_prism(
            view_map["view_0_front"],
            X,
            Z,
            Y,
            "XZ",
            use_holes=use_holes,
            facts=facts,
            skip_facts_cyl_holes=skip_facts_cyl_holes,
        )
        any_sig_holes = any_sig_holes or hh
        if p is not None:
            prisms.append(p)
    if "view_3_right" in view_map:
        p, hh = _extrude_view_prism(
            view_map["view_3_right"],
            Y,
            Z,
            X,
            "YZ",
            use_holes=use_holes,
            facts=facts,
            skip_facts_cyl_holes=skip_facts_cyl_holes,
        )
        any_sig_holes = any_sig_holes or hh
        if p is not None:
            prisms.append(p)
    elif "view_2_left" in view_map:
        p, hh = _extrude_view_prism(
            view_map["view_2_left"],
            Y,
            Z,
            X,
            "YZ",
            flip_horizontal=True,
            use_holes=use_holes,
            facts=facts,
            skip_facts_cyl_holes=skip_facts_cyl_holes,
        )
        any_sig_holes = any_sig_holes or hh
        if p is not None:
            prisms.append(p)
    if "view_4_top" in view_map:
        p, hh = _extrude_view_prism(
            view_map["view_4_top"],
            X,
            Y,
            Z,
            "XY",
            use_holes=use_holes,
            facts=facts,
            skip_facts_cyl_holes=skip_facts_cyl_holes,
        )
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


def _carve_iso_view(
    solid: cq.Shape,
    view_path: Path,
    view_name: str,
    diag: float,
    min_vol: float,
) -> cq.Shape:
    """Trim outer visual-hull overhang corners against a smooth isometric camera silhouette."""
    mesh = tessellate_shape(solid.wrapped, diag_mm=diag)
    if len(mesh.verts) == 0:
        return solid
    r_tuple, u_tuple, f_tuple = VIEW_CAMERAS[view_name]
    r_vec = np.asarray(r_tuple, dtype=np.float64)
    u_vec = np.asarray(u_tuple, dtype=np.float64)
    f_vec = np.asarray(f_tuple, dtype=np.float64)
    px = mesh.verts @ r_vec
    py = mesh.verts @ u_vec
    pz = mesh.verts @ f_vec
    x_min, x_max = float(px.min()), float(px.max())
    y_min, y_max = float(py.min()), float(py.max())
    z_min, z_max = float(pz.min()), float(pz.max())
    span_x = x_max - x_min
    span_y = y_max - y_min
    depth = (z_max - z_min) + 2.0
    if span_x < 1e-3 or span_y < 1e-3:
        return solid

    rgb = np.asarray(Image.open(view_path).convert("RGB"))
    mask = _erode_1px(np.any(rgb < 245, axis=-1))
    outers, _ = _extract_mask_loops(mask, grid_res=160, eps=0.0035)
    origin = x_min * r_vec + y_min * u_vec + (z_min - 1.0) * f_vec
    plane = cq.Plane(origin=tuple(origin.tolist()), xDir=r_tuple, normal=f_tuple)

    def make_wp() -> cq.Workplane:
        return cq.Workplane(plane)

    outer_solids: list[cq.Shape] = []
    for loop in outers:
        pts = [(u * span_x, v * span_y) for (u, v) in loop]
        pr = _extrude_loop_solid(make_wp, pts, depth, is_hole=False)
        if pr is not None and pr.isValid():
            outer_solids.append(pr)
    if not outer_solids:
        return solid

    prism = outer_solids[0]
    for extra in outer_solids[1:]:
        try:
            prism = prism.fuse(extra)
        except Exception:
            pass
    try:
        inter = solid.intersect(prism)
        solids = inter.Solids()
        if solids:
            top_s = max(solids, key=lambda s: s.Volume())
            if top_s.isValid() and top_s.Volume() > min_vol:
                return top_s
    except Exception:
        pass
    return solid


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
    min_vol = 0.02 * bounds_box.Volume()
    candidates: list[tuple[bool, cq.Shape]] = []

    rev = _build_revolve_candidate(view_map, X, Y, Z)
    if rev is not None:
        candidates.append((True, _cut_facts_bores(rev, facts, X, Y, Z)))

    tg = (facts or {}).get("through_holes_genus")
    has_facts_bores = bool((facts or {}).get("cylindrical_holes_and_bores"))
    primary_base: cq.Shape | None = None

    if tg is None or int(tg) > 0:
        hull_holes, has_sig_holes = _build_prism_hull(
            view_map, X, Y, Z, bounds_box, use_holes=True, facts=facts, skip_facts_cyl_holes=False
        )
        primary_base = hull_holes
        candidates.append((False, _cut_facts_bores(hull_holes, facts, X, Y, Z)))
        if has_sig_holes and has_facts_bores:
            hull_skipped, _ = _build_prism_hull(
                view_map, X, Y, Z, bounds_box, use_holes=True, facts=facts, skip_facts_cyl_holes=True
            )
            candidates.append((False, _cut_facts_bores(hull_skipped, facts, X, Y, Z)))
        if has_sig_holes:
            hull_plain, _ = _build_prism_hull(
                view_map, X, Y, Z, bounds_box, use_holes=False, facts=facts
            )
            candidates.append((False, _cut_facts_bores(hull_plain, facts, X, Y, Z)))
    else:
        hull_plain, _ = _build_prism_hull(
            view_map, X, Y, Z, bounds_box, use_holes=False, facts=facts
        )
        primary_base = hull_plain
        candidates.append((False, _cut_facts_bores(hull_plain, facts, X, Y, Z)))

    # Selectively add an iso-carved candidate only when isometric views trim meaningful (>2%) hull volume
    if primary_base is not None and ("view_6_iso_front_right_top" in view_map or "view_7_iso_back_left_bottom" in view_map):
        carved = primary_base
        if "view_6_iso_front_right_top" in view_map:
            carved = _carve_iso_view(carved, view_map["view_6_iso_front_right_top"], "view_6_iso_front_right_top", diag, min_vol)
        if "view_7_iso_back_left_bottom" in view_map:
            carved = _carve_iso_view(carved, view_map["view_7_iso_back_left_bottom"], "view_7_iso_back_left_bottom", diag, min_vol)
        if carved is not primary_base and carved.Volume() <= 0.98 * primary_base.Volume():
            candidates.append((False, _cut_facts_bores(carved, facts, X, Y, Z)))

    valid_cands: list[tuple[bool, cq.Shape, Any]] = []
    for is_rev, cand in candidates:
        try:
            tel = inspect_shape(cand.wrapped)
            if tel.is_valid and tel.solids == 1 and tel.shells == 1:
                valid_cands.append((is_rev, cand, tel))
        except Exception:
            continue

    best_solid: cq.Shape
    if len(valid_cands) == 1:
        best_solid = valid_cands[0][1]
    elif len(valid_cands) > 1:
        scored: list[tuple[float, cq.Shape]] = []
        for is_rev, cand, tel in valid_cands:
            try:
                diff = compare_against_facts(tel, facts)
                ve = evaluate_views(tessellate_shape(cand.wrapped, diag_mm=diag), views, eval_size=192)
                sc = score_candidate(tel, facts, diff, ve)
                # Prefer smooth axisymmetric revolved BRep over planar approximations when silhouettes clearly match
                rank = sc.total + (0.04 if is_rev and ve.mean_silhouette_iou >= 0.96 else 0.0)
                # Mild clean-BRep tiebreaker so unnecessary over-fragmentation never wins on 0.001 noise
                rank -= min(0.008, 0.00001 * tel.faces)
                scored.append((rank, cand))
            except Exception:
                continue
        best_solid = max(scored, key=lambda x: x[0])[1] if scored else valid_cands[0][1]
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
