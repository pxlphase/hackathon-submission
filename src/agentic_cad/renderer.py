from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GCPnts import GCPnts_UniformDeflection
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Camera definitions matching README.md:
# Each entry maps view_name -> (right_vec, up_vec, toward_viewer_vec)
# right x up == toward_viewer_vec (right-handed camera frame)
_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)
_SQRT6 = math.sqrt(6.0)

VIEW_CAMERAS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = {
    "view_0_front": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    "view_1_back": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "view_2_left": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
    "view_3_right": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "view_4_top": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "view_5_bottom": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "view_6_iso_front_right_top": (
        (1.0 / _SQRT2, 1.0 / _SQRT2, 0.0),
        (-1.0 / _SQRT6, 1.0 / _SQRT6, 2.0 / _SQRT6),
        (1.0 / _SQRT3, -1.0 / _SQRT3, 1.0 / _SQRT3),
    ),
    "view_7_iso_back_left_bottom": (
        (-1.0 / _SQRT2, -1.0 / _SQRT2, 0.0),
        (-1.0 / _SQRT6, 1.0 / _SQRT6, 2.0 / _SQRT6),
        (-1.0 / _SQRT3, 1.0 / _SQRT3, -1.0 / _SQRT3),
    ),
}


@dataclass
class TessellatedModel:
    verts: np.ndarray  # (N, 3) float64
    tris: np.ndarray  # (T, 3) int32
    tri_normals: np.ndarray  # (T, 3) float64
    edge_polylines: list[np.ndarray]  # list of (K, 3) float64 arrays


@dataclass
class RenderedView:
    name: str
    image: Image.Image
    mask: np.ndarray  # bool (H, W)
    edge_mask: np.ndarray  # bool (H, W)
    aspect_wh: float


@dataclass
class ViewMetric:
    name: str
    target_path: Path
    silhouette_iou: float
    edge_f1: float
    flipped_silhouette_iou: float
    aspect_err: float


@dataclass
class ViewEvalResult:
    metrics: list[ViewMetric]
    mean_silhouette_iou: float
    min_silhouette_iou: float
    mean_edge_f1: float
    handedness_flip_detected: bool
    rendered_views: dict[str, RenderedView]

    @property
    def worst_views(self) -> list[ViewMetric]:
        return sorted(
            self.metrics,
            key=lambda m: (0.75 * m.silhouette_iou + 0.25 * m.edge_f1),
        )


def tessellate_shape(shape: TopoDS_Shape, diag_mm: float | None = None) -> TessellatedModel:
    """Tessellate a TopoDS_Shape into triangle mesh and discretized BRep crease polylines."""
    if diag_mm is None or diag_mm <= 0:
        diag_mm = 100.0
    lin_defl = max(0.04, min(1.5, 0.002 * diag_mm))
    BRepMesh_IncrementalMesh(shape, lin_defl, False, 0.35, True)

    verts_list: list[np.ndarray] = []
    tris_list: list[np.ndarray] = []
    v_offset = 0

    fmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
    for idx in range(1, fmap.Extent() + 1):
        face = TopoDS.Face_s(fmap.FindKey(idx))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None or tri.NbNodes() == 0 or tri.NbTriangles() == 0:
            continue
        trsf = loc.Transformation()
        nv = tri.NbNodes()
        nt = tri.NbTriangles()
        pts = np.empty((nv, 3), dtype=np.float64)
        for k in range(1, nv + 1):
            p = tri.Node(k).Transformed(trsf)
            pts[k - 1] = (p.X(), p.Y(), p.Z())
        ts = np.empty((nt, 3), dtype=np.int32)
        rev = face.Orientation() == TopAbs_REVERSED
        for k in range(1, nt + 1):
            n1, n2, n3 = tri.Triangle(k).Get()
            if rev:
                ts[k - 1] = (n1 - 1 + v_offset, n3 - 1 + v_offset, n2 - 1 + v_offset)
            else:
                ts[k - 1] = (n1 - 1 + v_offset, n2 - 1 + v_offset, n3 - 1 + v_offset)
        verts_list.append(pts)
        tris_list.append(ts)
        v_offset += nv

    if verts_list:
        verts = np.vstack(verts_list)
        tris = np.vstack(tris_list)
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        tri_normals = cross / norms
    else:
        verts = np.zeros((0, 3), dtype=np.float64)
        tris = np.zeros((0, 3), dtype=np.int32)
        tri_normals = np.zeros((0, 3), dtype=np.float64)

    edge_polylines: list[np.ndarray] = []
    emap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, emap)
    edge_defl = max(0.05, min(2.0, 0.0025 * diag_mm))
    for idx in range(1, emap.Extent() + 1):
        edge = TopoDS.Edge_s(emap.FindKey(idx))
        if BRep_Tool.Degenerated_s(edge):
            continue
        try:
            curve = BRepAdaptor_Curve(edge)
            algo = GCPnts_UniformDeflection(curve, edge_defl)
            if algo.IsDone() and algo.NbPoints() >= 2:
                npts = algo.NbPoints()
                arr = np.empty((npts, 3), dtype=np.float64)
                for p_idx in range(1, npts + 1):
                    p = algo.Value(p_idx)
                    arr[p_idx - 1] = (p.X(), p.Y(), p.Z())
                edge_polylines.append(arr)
        except Exception:
            continue

    return TessellatedModel(verts=verts, tris=tris, tri_normals=tri_normals, edge_polylines=edge_polylines)


def render_view(model: TessellatedModel, view_name: str, size: int = 384) -> RenderedView:
    """Render an orthographic view using fast C-backed triangle painting + depth-tested BRep edges."""
    if view_name not in VIEW_CAMERAS:
        raise ValueError(f"Unknown camera view: {view_name}")
    r_tuple, u_tuple, f_tuple = VIEW_CAMERAS[view_name]
    r_vec = np.asarray(r_tuple, dtype=np.float64)
    u_vec = np.asarray(u_tuple, dtype=np.float64)
    f_vec = np.asarray(f_tuple, dtype=np.float64)

    if len(model.verts) == 0 or len(model.tris) == 0:
        blank = Image.new("RGB", (size, size), (255, 255, 255))
        return RenderedView(
            name=view_name,
            image=blank,
            mask=np.zeros((size, size), dtype=bool),
            edge_mask=np.zeros((size, size), dtype=bool),
            aspect_wh=1.0,
        )

    px = model.verts @ r_vec
    py = model.verts @ u_vec
    pz = model.verts @ f_vec

    x_min, x_max = float(px.min()), float(px.max())
    y_min, y_max = float(py.min()), float(py.max())
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    max_span = max(span_x, span_y)
    aspect_wh = span_x / span_y

    # Match ~96.2% tight auto-fit framing of input view PNGs
    scale = (size * 0.96) / max_span
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    sx = (px - cx) * scale + 0.5 * size
    sy = -(py - cy) * scale + 0.5 * size

    # Triangle depths & lighting (simple CAD diffuse matching tan/brown target theme)
    t_sx = sx[model.tris]
    t_sy = sy[model.tris]
    t_z = pz[model.tris].mean(axis=1)
    # Key light from upper-left-viewer
    light_dir = 0.72 * f_vec + 0.42 * u_vec - 0.35 * r_vec
    light_dir /= np.linalg.norm(light_dir)
    ndotl = np.abs(model.tri_normals @ light_dir)
    shade = np.clip(0.38 + 0.62 * ndotl, 0.25, 1.0)

    # Sort back-to-front by centroid depth
    order = np.argsort(t_z)

    # Encode 15-bit depth into (R, G) alongside luminance in B for single-pass C rasterization
    z_lo, z_hi = float(t_z.min()), float(t_z.max())
    z_span = max(z_hi - z_lo, 1e-6)
    z_norm = np.clip((t_z - z_lo) / z_span, 0.0, 1.0)
    z_u15 = np.clip((1.0 + z_norm * 32000.0).astype(np.int32), 1, 32767)

    # Image channels: R = z_hi_byte, G = z_lo_byte, B = shade_byte; background = (0, 0, 0)
    buf_img = Image.new("RGB", (size, size), (0, 0, 0))
    draw_buf = ImageDraw.Draw(buf_img)

    # Skip degenerate sub-pixel slivers for speed while guaranteeing full silhouette
    area2 = (t_sx[:, 1] - t_sx[:, 0]) * (t_sy[:, 2] - t_sy[:, 0]) - (t_sy[:, 1] - t_sy[:, 0]) * (
        t_sx[:, 2] - t_sx[:, 0]
    )
    valid_idx = order[np.abs(area2[order]) > 0.01]

    t_sx_list = t_sx[valid_idx].tolist()
    t_sy_list = t_sy[valid_idx].tolist()
    z_list = z_u15[valid_idx].tolist()
    s_list = np.clip(shade[valid_idx] * 255.0, 40, 255).astype(np.uint8).tolist()

    for xs, ys, zv, sv in zip(t_sx_list, t_sy_list, z_list, s_list):
        draw_buf.polygon(
            [(xs[0], ys[0]), (xs[1], ys[1]), (xs[2], ys[2])],
            fill=(zv >> 8, zv & 0xFF, sv),
        )

    arr = np.asarray(buf_img)
    depth_buf = (arr[:, :, 0].astype(np.int32) << 8) | arr[:, :, 1].astype(np.int32)
    mask = depth_buf > 0
    lum = arr[:, :, 2].astype(np.float32) / 255.0

    # Colorize tan/brown CAD palette on pure white background
    rgb = np.full((size, size, 3), 255, dtype=np.uint8)
    base_color = np.array([196.0, 154.0, 98.0], dtype=np.float32)
    shaded_rgb = np.clip(lum[:, :, None] * base_color[None, None, :], 0, 255).astype(np.uint8)
    rgb[mask] = shaded_rgb[mask]

    # Fast 3x3 min-depth dilation so visible silhouette/corner edges aren't occluded by neighbor depth
    pad_d = np.pad(depth_buf, 1, mode="constant", constant_values=0)
    local_min = np.minimum.reduce(
        [
            pad_d[0:size, 0:size],
            pad_d[0:size, 1 : size + 1],
            pad_d[0:size, 2 : size + 2],
            pad_d[1 : size + 1, 0:size],
            pad_d[1 : size + 1, 1 : size + 1],
            pad_d[1 : size + 1, 2 : size + 2],
            pad_d[2 : size + 2, 0:size],
            pad_d[2 : size + 2, 1 : size + 1],
            pad_d[2 : size + 2, 2 : size + 2],
        ]
    )
    depth_tol = 450  # ~1.4% of quantized depth range

    edge_mask = np.zeros((size, size), dtype=bool)
    for poly in model.edge_polylines:
        ex = (poly @ r_vec - cx) * scale + 0.5 * size
        ey = -((poly @ u_vec - cy) * scale) + 0.5 * size
        ez = poly @ f_vec
        ez_u15 = np.clip((1.0 + np.clip((ez - z_lo) / z_span, 0.0, 1.0) * 32000.0), 1, 32767)
        for j in range(len(poly) - 1):
            x0, y0, z0 = ex[j], ey[j], ez_u15[j]
            x1, y1, z1 = ex[j + 1], ey[j + 1], ez_u15[j + 1]
            dist = max(abs(x1 - x0), abs(y1 - y0))
            steps = max(2, min(int(math.ceil(dist)) + 1, 512))
            ts = np.linspace(0.0, 1.0, steps)
            ix = np.clip(np.rint(x0 + (x1 - x0) * ts).astype(np.int32), 0, size - 1)
            iy = np.clip(np.rint(y0 + (y1 - y0) * ts).astype(np.int32), 0, size - 1)
            iz = z0 + (z1 - z0) * ts
            vis = (iz + depth_tol) >= local_min[iy, ix]
            if np.any(vis):
                edge_mask[iy[vis], ix[vis]] = True

    rgb[edge_mask] = np.array([25, 25, 25], dtype=np.uint8)
    return RenderedView(
        name=view_name,
        image=Image.fromarray(rgb, mode="RGB"),
        mask=mask,
        edge_mask=edge_mask,
        aspect_wh=aspect_wh,
    )


def _normalize_mask_box(mask: np.ndarray, edge_mask: np.ndarray, out_size: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Tightly crop mask+edges to non-zero bbox and center inside out_size preserving aspect ratio."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        empty = np.zeros((out_size, out_size), dtype=bool)
        return empty, empty, 1.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    aspect = w / max(1, h)
    scale = (out_size * 0.92) / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    crop_m = Image.fromarray((mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255), mode="L")
    crop_e = Image.fromarray((edge_mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8) * 255), mode="L")
    res_m = crop_m.resize((nw, nh), Image.Resampling.NEAREST)
    res_e = crop_e.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas_m = Image.new("L", (out_size, out_size), 0)
    canvas_e = Image.new("L", (out_size, out_size), 0)
    ox = (out_size - nw) // 2
    oy = (out_size - nh) // 2
    canvas_m.paste(res_m, (ox, oy))
    canvas_e.paste(res_e, (ox, oy))
    return np.asarray(canvas_m) > 127, np.asarray(canvas_e) > 64, aspect


def _dilate_bool(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(mask)
            src_y = slice(max(0, -dy), mask.shape[0] - max(0, dy))
            dst_y = slice(max(0, dy), mask.shape[0] - max(0, -dy))
            src_x = slice(max(0, -dx), mask.shape[1] - max(0, dx))
            dst_x = slice(max(0, dx), mask.shape[1] - max(0, -dx))
            shifted[dst_y, dst_x] = mask[src_y, src_x]
            out |= shifted
    return out


def _erode_1px(mask: np.ndarray) -> np.ndarray:
    """Strip 1-px outer boundary stroke from high-res organiser orthographic renders."""
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


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / union) if union > 0 else 0.0


def _extract_target_edge_mask(tgt_im: Image.Image, tgt_rgb: np.ndarray) -> np.ndarray:
    """Extract sharp dark BRep crease lines & outline without misclassifying dark shaded faces."""
    fg = np.any(tgt_rgb < 245, axis=-1)
    max_rgb = np.max(tgt_rgb, axis=-1)
    very_dark = max_rgb < 55
    lum = np.asarray(tgt_im.convert("L"), dtype=np.int16)
    local_hi = np.asarray(tgt_im.convert("L").filter(ImageFilter.MaxFilter(5)), dtype=np.int16)
    local_contrast = (local_hi - lum >= 32) & (lum < 120)
    return fg & (very_dark | local_contrast)


def evaluate_views(
    model: TessellatedModel,
    target_views: list[Path],
    eval_size: int = 320,
) -> ViewEvalResult:
    """Render candidate across target camera views and measure Silhouette IoU, Edge F1 & handedness."""
    metrics: list[ViewMetric] = []
    rendered_views: dict[str, RenderedView] = {}

    for path in target_views:
        view_name = path.stem
        if view_name not in VIEW_CAMERAS:
            continue
        tgt_im = Image.open(path).convert("RGB")
        tgt_rgb = np.asarray(tgt_im)
        tgt_mask_raw = _erode_1px(np.any(tgt_rgb < 245, axis=-1))
        # Extract sharp dark BRep crease lines + outer boundary before normalization
        tgt_dark = _extract_target_edge_mask(tgt_im, tgt_rgb)
        tgt_mask, tgt_edge, tgt_aspect = _normalize_mask_box(tgt_mask_raw, tgt_dark, eval_size)

        rv = render_view(model, view_name, size=eval_size)
        rendered_views[view_name] = rv
        gen_mask, gen_edge, gen_aspect = _normalize_mask_box(rv.mask, rv.edge_mask, eval_size)

        iou_orig = _iou(tgt_mask, gen_mask)
        iou_flip = _iou(tgt_mask, gen_mask[:, ::-1])

        # Tolerant dilated edge F1 (catches blind pockets / internal cutouts)
        tgt_e_dil = _dilate_bool(tgt_edge, 2)
        gen_e_dil = _dilate_bool(gen_edge, 2)
        prec = float(np.logical_and(gen_edge, tgt_e_dil).sum() / max(1, int(gen_edge.sum())))
        rec = float(np.logical_and(tgt_edge, gen_e_dil).sum() / max(1, int(tgt_edge.sum())))
        edge_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        aspect_err = abs(gen_aspect - tgt_aspect) / max(tgt_aspect, 1e-4)
        metrics.append(
            ViewMetric(
                name=view_name,
                target_path=path,
                silhouette_iou=round(iou_orig, 4),
                edge_f1=round(edge_f1, 4),
                flipped_silhouette_iou=round(iou_flip, 4),
                aspect_err=round(aspect_err, 4),
            )
        )

    if not metrics:
        return ViewEvalResult(
            metrics=[],
            mean_silhouette_iou=0.0,
            min_silhouette_iou=0.0,
            mean_edge_f1=0.0,
            handedness_flip_detected=False,
            rendered_views=rendered_views,
        )

    mean_iou = sum(m.silhouette_iou for m in metrics) / len(metrics)
    min_iou = min(m.silhouette_iou for m in metrics)
    mean_edge = sum(m.edge_f1 for m in metrics) / len(metrics)

    # Detect consistent horizontal mirror error across asymmetric views
    flip_wins = sum(1 for m in metrics if m.flipped_silhouette_iou > m.silhouette_iou + 0.06)
    handedness_flip = flip_wins >= 2

    return ViewEvalResult(
        metrics=metrics,
        mean_silhouette_iou=round(mean_iou, 4),
        min_silhouette_iou=round(min_iou, 4),
        mean_edge_f1=round(mean_edge, 4),
        handedness_flip_detected=handedness_flip,
        rendered_views=rendered_views,
    )


def build_visual_diff_composite(
    eval_result: ViewEvalResult,
    worst_n: int = 2,
    tile_px: int = 384,
) -> bytes | None:
    """Build one compact image showing [Target | Generated | Red=Missing / Blue=Extra] for worst views."""
    worst = eval_result.worst_views[:worst_n]
    if not worst:
        return None

    header_h = 24
    cols = 3  # Target | Generated | Red-Blue Diff
    rows = len(worst)
    canvas = Image.new("RGB", (cols * tile_px, rows * (tile_px + header_h)), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for r_idx, metric in enumerate(worst):
        y0 = r_idx * (tile_px + header_h)
        rv = eval_result.rendered_views.get(metric.name)
        if rv is None:
            continue

        tgt_raw_im = Image.open(metric.target_path).convert("RGB")
        tgt_raw_arr = np.asarray(tgt_raw_im)
        tgt_im = tgt_raw_im.resize((tile_px, tile_px), Image.Resampling.LANCZOS)
        gen_im = rv.image.resize((tile_px, tile_px), Image.Resampling.LANCZOS)

        # Normalized red/blue silhouette + orange missing-internal-edge difference overlay
        tgt_mask_raw = _erode_1px(np.any(tgt_raw_arr < 245, axis=-1))
        tgt_edge_raw = _extract_target_edge_mask(tgt_raw_im, tgt_raw_arr)
        t_norm, t_edge_norm, _ = _normalize_mask_box(tgt_mask_raw, tgt_edge_raw, tile_px)
        g_norm, g_edge_norm, _ = _normalize_mask_box(rv.mask, rv.edge_mask, tile_px)

        diff_arr = np.full((tile_px, tile_px, 3), 255, dtype=np.uint8)
        both = t_norm & g_norm
        missing = t_norm & ~g_norm
        extra = g_norm & ~t_norm
        missing_internal_edges = both & t_edge_norm & ~_dilate_bool(g_edge_norm, 2)
        diff_arr[both] = [145, 130, 155]  # Neutral gray-purple overlap
        diff_arr[missing_internal_edges] = [255, 165, 0]  # Orange = missing internal pocket/bore/crease edge
        diff_arr[missing] = [225, 45, 45]  # Red = target silhouette missing
        diff_arr[extra] = [40, 105, 235]  # Blue = extra unwanted material
        diff_im = Image.fromarray(diff_arr, mode="RGB")

        labels = [
            f"TARGET: {metric.name}",
            f"GENERATED (IoU={metric.silhouette_iou:.2f}, Edge={metric.edge_f1:.2f})",
            "DIFF (RED=Miss, BLUE=Extra, ORANGE=MissEdge)",
        ]
        for c_idx, (lbl, tile) in enumerate(zip(labels, (tgt_im, gen_im, diff_im))):
            x0 = c_idx * tile_px
            draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + header_h - 1], fill=(32, 36, 44))
            draw.text((x0 + 6, y0 + 5), lbl, fill=(245, 245, 245), font=font)
            canvas.paste(tile, (x0, y0 + header_h))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
