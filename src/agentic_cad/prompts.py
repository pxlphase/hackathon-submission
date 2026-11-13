from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agentic_cad.cadquery_ref import CADQUERY_CHEATSHEET

_VIEW_TITLES: dict[str, str] = {
    "view_0_front": "VIEW 0: FRONT (+Y) [X right, Z up]",
    "view_1_back": "VIEW 1: BACK (-Y) [X left, Z up]",
    "view_2_left": "VIEW 2: LEFT (+X) [Y left, Z up]",
    "view_3_right": "VIEW 3: RIGHT (-X) [Y right, Z up]",
    "view_4_top": "VIEW 4: TOP (-Z) [X right, Y up]",
    "view_5_bottom": "VIEW 5: BOTTOM (+Z) [X right, Y down]",
    "view_6_iso_front_right_top": "VIEW 6: ISO (+X, -Y, +Z)",
    "view_7_iso_back_left_bottom": "VIEW 7: ISO (-X, +Y, -Z)",
}

SYSTEM_PROMPT = f"""You are an expert mechanical CAD engineer writing exact, production-ready CadQuery 2.8 Python scripts.
Your task is to reconstruct a mechanical solid 1:1 from its engineering specification (`description.md`), ground-truth BRep measurements (`facts.json`), and 8 orthographic/isometric views.

{CADQUERY_CHEATSHEET}

## Strategy for High Geometric Accuracy
1. Always align your model to the canonical `[0..X, 0..Y, 0..Z]` bounding-box frame (`min=(0,0,0), max=(X,Y,Z)`).
2. Double-check feature axis, center coordinates, blind vs through depths, and handedness against BOTH `facts.json` and ALL 8 views.
3. For symmetric parts or polar/angled banks, compute exact center offsets directly in `[0..X, 0..Y, 0..Z]`.
4. Be decisive and concise in reasoning: derive coordinates cleanly from `facts.json` and the brief without repetitive re-derivations, and output ONLY one complete executable Python script inside a ```python ... ``` code fence assigning the single solid to `result`.
"""


def _png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_contact_sheets(views: list[Path], tile_px: int = 512) -> list[bytes]:
    """Pack up to 8 orthographic/isometric views into two labelled 2x2 contact sheets (4x fewer vision tokens)."""
    view_map = {p.stem: p for p in views}
    groups = [
        ["view_0_front", "view_1_back", "view_2_left", "view_3_right"],
        ["view_4_top", "view_5_bottom", "view_6_iso_front_right_top", "view_7_iso_back_left_bottom"],
    ]

    header_h = 24
    inner_h = tile_px - header_h
    font = ImageFont.load_default()
    sheets: list[bytes] = []

    for grp in groups:
        present = [name for name in grp if name in view_map]
        if not present:
            continue
        sheet = Image.new("RGB", (tile_px * 2, tile_px * 2), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)

        for idx, name in enumerate(grp):
            col = idx % 2
            row = idx // 2
            x0 = col * tile_px
            y0 = row * tile_px
            draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + header_h - 1], fill=(28, 34, 42))
            label = _VIEW_TITLES.get(name, name)
            draw.text((x0 + 8, y0 + 6), label, fill=(245, 245, 245), font=font)

            if name in view_map:
                im = Image.open(view_map[name]).convert("RGB")
                # Aspect-preserving resize inside (tile_px - 8, inner_h - 8)
                im.thumbnail((tile_px - 8, inner_h - 8), Image.Resampling.LANCZOS)
                px = x0 + (tile_px - im.width) // 2
                py = y0 + header_h + (inner_h - im.height) // 2
                sheet.paste(im, (px, py))

            # Subtle grid separator
            draw.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1], outline=(200, 205, 212), width=1)

        buf = io.BytesIO()
        sheet.save(buf, format="PNG", optimize=True)
        sheets.append(buf.getvalue())

    # If non-standard view names existed outside the 8 expected names, append them directly
    known = {n for g in groups for n in g}
    extras = [p for p in views if p.stem not in known]
    for p in extras[:2]:
        im = Image.open(p).convert("RGB")
        im.thumbnail((tile_px, tile_px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        sheets.append(buf.getvalue())

    return sheets


def resize_context_image(context_path: Path, max_px: int = 512) -> bytes:
    im = Image.open(context_path).convert("RGB")
    im.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def format_facts_brief(facts: dict[str, Any] | None) -> str:
    """Serialize facts.json concisely without losing any millimeter measurements."""
    if not facts:
        return "(facts.json not provided; rely on description.md and the 8 views.)"

    lines: list[str] = ["### Ground-Truth Telemetry (`facts.json`, origin = bbox min corner `[0..X, 0..Y, 0..Z]`):"]
    bbox = facts.get("bounding_box_mm", {})
    lines.append(
        f"- **Bounding box (mm)**: X={bbox.get('x')}, Y={bbox.get('y')}, Z={bbox.get('z')}"
    )
    lines.append(
        f"- **Mass/Topology**: volume={facts.get('volume_mm3')} mm^3, "
        f"surface_area={facts.get('surface_area_mm2')} mm^2, "
        f"solids={facts.get('solids')}, through_holes_genus={facts.get('through_holes_genus')}, "
        f"faces={facts.get('faces')}, edges={facts.get('edges')}"
    )
    if facts.get("face_types"):
        ft_str = ", ".join(f"{k}={v}" for k, v in facts["face_types"].items())
        lines.append(f"- **Face types**: {ft_str}")

    def fmt_cyl_list(title: str, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        lines.append(f"- **{title}**:")
        for g in entries:
            centres = ", ".join(str([round(c, 2) for c in ctr]) for ctr in g.get("centres", []))
            lengths = g.get("lengths", [])
            extra = f" (+{g['more_not_listed']} more in description.md)" if g.get("more_not_listed") else ""
            lines.append(
                f"  * count={g.get('count')}, dia={g.get('diameter')}mm, axis={g.get('axis')}, "
                f"lengths={lengths}, centres=[{centres}]{extra}"
            )

    fmt_cyl_list("Cylindrical holes & bores", facts.get("cylindrical_holes_and_bores", []))
    fmt_cyl_list("Cylindrical shafts & bosses", facts.get("cylindrical_shafts_and_bosses", []))

    if facts.get("partial_cylinders_arcs"):
        arcs = "; ".join(
            f"dia={a.get('diameter')}mm ({a.get('type')}, faces={a.get('faces')})"
            for a in facts["partial_cylinders_arcs"]
        )
        lines.append(f"- **Partial cylinder arcs**: {arcs}")

    if facts.get("fillets_rounds_from_cylinders"):
        rounds = "; ".join(
            f"R={r.get('radius')}mm ({r.get('type')}, faces={r.get('faces')})"
            for r in facts["fillets_rounds_from_cylinders"]
        )
        lines.append(f"- **Cylinder rounds/fillets**: {rounds}")

    if facts.get("fillets_from_tori_minor_radii"):
        lines.append(f"- **Torus fillet minor radii (mm -> faces)**: {facts['fillets_from_tori_minor_radii']}")

    if facts.get("cones_half_angle_deg"):
        lines.append(f"- **Cones half-angle deg -> faces**: {facts['cones_half_angle_deg']}")

    return "\n".join(lines)


def _erode_1px(mask: np.ndarray) -> np.ndarray:
    """Trim 1-px outer stroke on high-res orthographic renders so mm spans reflect geometry."""
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


def _extract_ortho_envelope_profiles(views: list[Path], facts: dict[str, Any] | None) -> str:
    """Sample 7 orthographic silhouette stations in canonical [0..X, 0..Y, 0..Z] mm coordinates."""
    if not facts or not views:
        return ""
    bbox = facts.get("bounding_box_mm", {})
    try:
        X = float(bbox.get("x") or 0.0)
        Y = float(bbox.get("y") or 0.0)
        Z = float(bbox.get("z") or 0.0)
    except (TypeError, ValueError):
        return ""
    if min(X, Y, Z) <= 0.1:
        return ""

    vmap = {p.stem: p for p in views}
    # Camera orientations match VIEW_CAMERAS:
    # view_1_back has right=-X, up=+Z -> horizontal flip matches view_0_front
    # view_2_left has right=-Y, up=+Z -> horizontal flip matches view_3_right
    # view_5_bottom has right=+X, up=-Y -> vertical flip matches view_4_top
    specs = [
        ("Front XZ", "view_0_front", False, "view_1_back", True, False, "Z", Z, "X", X),
        ("Right YZ", "view_3_right", False, "view_2_left", True, False, "Z", Z, "Y", Y),
        ("Top XY", "view_4_top", False, "view_5_bottom", False, True, "Y", Y, "X", X),
    ]
    stations = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
    lines: list[str] = []

    for label, p_name, p_fh, f_name, f_fh, f_fv, v_ax, v_len, h_ax, h_len in specs:
        vp = vmap.get(p_name)
        fh, fv = p_fh, False
        if vp is None:
            vp = vmap.get(f_name)
            fh, fv = f_fh, f_fv
        if vp is None or not vp.is_file():
            continue
        try:
            rgb = np.asarray(Image.open(vp).convert("RGB"))
        except Exception:
            continue
        mask = _erode_1px(np.any(rgb < 245, axis=-1))
        if fh:
            mask = mask[:, ::-1]
        if fv:
            mask = mask[::-1, :]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        crop = mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        H, W = crop.shape
        min_gap_px = max(4, int(0.03 * W))
        denom = max(1, W - 1)
        entries: list[str] = []
        for s in stations:
            r = min(H - 1, max(0, int(round((1.0 - s) * (H - 1)))))
            cols = np.where(crop[r, :])[0]
            v_mm = s * v_len
            if len(cols) == 0:
                entries.append(f"{v_ax}={v_mm:.1f}->empty")
                continue
            breaks = np.where(np.diff(cols) > min_gap_px)[0]
            starts = np.r_[cols[0], cols[breaks + 1]]
            ends = np.r_[cols[breaks], cols[-1]]
            runs = list(zip(starts.tolist(), ends.tolist()))
            if len(runs) <= 3:
                span = "+".join(f"[{a / denom * h_len:.1f}..{b / denom * h_len:.1f}]" for a, b in runs)
                entries.append(f"{v_ax}={v_mm:.1f}->{h_ax}{span}")
            else:
                lo = (int(cols[0]) / denom) * h_len
                hi = (int(cols[-1]) / denom) * h_len
                entries.append(f"{v_ax}={v_mm:.1f}->{h_ax}[{lo:.1f}..{hi:.1f}]")
        lines.append(f"  * {label}: " + "; ".join(entries))

    if not lines:
        return ""
    return (
        "### Orthographic Silhouette Outer Envelopes (measured from views in `[0..X, 0..Y, 0..Z]` mm):\n"
        + "\n".join(lines)
    )


def build_initial_user_content(
    case_name: str,
    description: str,
    facts: dict[str, Any] | None,
    views: list[Path],
    context: Path | None,
    tile_px: int = 512,
    context_px: int = 512,
    enable_tools: bool = False,
) -> list[dict[str, Any]]:
    """Construct the prefix-cacheable Turn-0 multimodal user message."""
    from agentic_cad.tools import summarize_ortho_key_features

    facts_text = format_facts_brief(facts)
    ortho_text = _extract_ortho_envelope_profiles(views, facts)
    key_feats_text = ""
    try:
        key_feats_text = summarize_ortho_key_features(views, facts)
    except Exception:
        key_feats_text = ""

    extra_sections = ""
    if ortho_text:
        extra_sections += f"\n\n{ortho_text}"
    if key_feats_text:
        extra_sections += f"\n\n{key_feats_text}"

    tool_section = ""
    if enable_tools:
        tool_section = (
            "\n\n### Optional Inspection & Probing Tools\n"
            "You have access to 3 tools if needed before outputting your final script:\n"
            "- `measure_ortho_features(view, detail)`: extracts full canonical outer/hole vertices and internal creases.\n"
            "- `zoom_view_crop(view, h_min, v_min, h_max, v_max)`: zooms native 1024px views with mm axis rulers.\n"
            "- `cad_probe(code)`: runs CadQuery safely and returns volume, bbox, missing features, and 8-view IoU.\n"
            "For straightforward parts, write the script directly or run one `cad_probe`; for complex/angled shapes, "
            "measure or zoom ambiguities first, probe cleanly, and finish with one ```python ... ``` block assigning `result`."
        )

    text_block = (
        f"## Component: {case_name}\n\n"
        f"### Engineering Design Brief (`description.md`)\n{description.strip()}\n\n"
        f"{facts_text}{extra_sections}{tool_section}\n\n"
        f"### Instructions\n"
        f"Inspect all 8 orthographic/isometric views in the two contact sheets below, verify every "
        f"position in `[0..X, 0..Y, 0..Z]`, and write a clean CadQuery script that assigns the "
        f"single solid to `result`."
    )

    content: list[dict[str, Any]] = [{"type": "text", "text": text_block}]
    for sheet_bytes in build_contact_sheets(views, tile_px=tile_px):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _png_bytes_to_data_url(sheet_bytes)},
            }
        )
    if context is not None and context.is_file():
        ctx_bytes = resize_context_image(context, max_px=context_px)
        content.append(
            {
                "type": "text",
                "text": "Reference `context.png` (target component highlighted in orange inside its assembly):",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _png_bytes_to_data_url(ctx_bytes)},
            }
        )
    return content


def build_repair_messages(
    initial_user_content: list[dict[str, Any]],
    last_code: str,
    diagnostic_report: str,
    diff_composite_png: bytes | None = None,
    enable_tools: bool = False,
) -> list[dict[str, Any]]:
    """Keep System + Initial User prefix identical across turns for vLLM KV prefix-cache reuse."""
    tool_hint = (
        " You may inspect discrepancies with `zoom_view_crop` / `measure_ortho_features` or verify a fix via `cad_probe` before answering."
        if enable_tools
        else ""
    )
    repair_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "### Automated Verification & Repair Feedback\n"
                f"{diagnostic_report.strip()}\n\n"
                "Fix ONLY the discrepancies identified above while keeping accurate dimensions/features intact. "
                f"Verify coordinate axes, workplane normals, hole diameters/centres, and boolean cut directions.{tool_hint} "
                "Output the complete corrected CadQuery Python script assigning the final solid to `result`."
            ),
        }
    ]
    if diff_composite_png is not None:
        repair_content.append(
            {
                "type": "text",
                "text": (
                    "Visual diff image below (Target | Generated | Diff): "
                    "RED = missing outer material, BLUE = extra unwanted material, "
                    "ORANGE = missing internal crease/pocket/bore edges."
                ),
            }
        )
        repair_content.append(
            {
                "type": "image_url",
                "image_url": {"url": _png_bytes_to_data_url(diff_composite_png)},
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_content},
        {"role": "assistant", "content": f"```python\n{last_code.strip()}\n```"},
        {"role": "user", "content": repair_content},
    ]
