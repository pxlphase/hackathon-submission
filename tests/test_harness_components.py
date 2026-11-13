from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cadquery as cq
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.main as cli_main
from agentic_cad.agent import QwenCadAgent
from agentic_cad.cadquery_ref import extract_python_code, validate_cadquery_ast
from agentic_cad.config import HarnessConfig, load_dotenv_if_present
from agentic_cad.executor import execute_cadquery_script, strip_fillets_and_chamfers
from agentic_cad.fallback import generate_fallback_step
from agentic_cad.inspector import compare_against_facts, inspect_shape, inspect_step_file, load_step_shape
from agentic_cad.prompts import (
    SYSTEM_PROMPT,
    build_contact_sheets,
    build_initial_user_content,
    build_repair_messages,
)
from agentic_cad.renderer import (
    VIEW_CAMERAS,
    _extract_target_edge_mask,
    build_visual_diff_composite,
    evaluate_views,
    render_view,
    tessellate_shape,
)
from agentic_cad.scorer import score_candidate


class _FakeCompletions:
    def __init__(self, handler):
        self._handler = handler
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._handler(len(self.calls) - 1, kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply, reasoning_content=None))]
        )


class _FakeOpenAIClient:
    def __init__(self, handler):
        self.completions = _FakeCompletions(handler)
        self.chat = SimpleNamespace(completions=self.completions)


def _test_config(**overrides) -> HarnessConfig:
    base = dict(
        openai_base_url="http://localhost:8000/v1",
        openai_api_key="test-key",
        openai_model="Qwen/Qwen3.8-27B",
        max_workers=2,
        global_timeout_s=120.0,
        exec_timeout_s=20.0,
        llm_timeout_s=30.0,
        contact_tile_px=256,
        context_max_px=256,
        diff_tile_px=192,
        render_eval_px=192,
    )
    base.update(overrides)
    return HarnessConfig(**base)


def test_load_test_cases_and_facts():
    inputs_dir = REPO_ROOT / "artifacts/inputs"
    cases = cli_main.load_test_cases(inputs_dir)
    assert len(cases) == 9
    assert [c.name for c in cases] == [f"test_case_{i}" for i in range(1, 10)]
    for tc in cases:
        assert len(tc.views) == 8
        assert tc.facts is not None
        assert "bounding_box_mm" in tc.facts
    by_name = {c.name: c for c in cases}
    assert by_name["test_case_1"].context is not None
    assert by_name["test_case_7"].context is None
    assert by_name["test_case_8"].context is None


def test_config_from_env_and_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'OPENAI_BASE_URL="http://10.0.0.5:8000/v1"\n'
        "OPENAI_API_KEY=secret-123\n"
        "CAD_MAX_WORKERS=3\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "pre-existing-wins")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CAD_MAX_WORKERS", raising=False)

    load_dotenv_if_present(env_file)
    cfg = HarnessConfig.from_env(load_dotenv=False)
    assert cfg.openai_base_url == "http://10.0.0.5:8000/v1"
    assert cfg.openai_api_key == "pre-existing-wins"
    assert cfg.max_workers == 3
    assert cfg.global_timeout_s == 3300.0


def test_extract_python_code_and_validate_ast():
    raw = """
<think>Let me analyze the plate dimensions carefully...</think>
Here is a draft snippet:
```python
x = 1
```
And here is the final solution:
```python
import cadquery as cq
result = cq.Workplane("XY").box(20, 10, 5, centered=(False, False, False))
```
"""
    extracted = extract_python_code(raw)
    assert "box(20, 10, 5" in extracted
    assert validate_cadquery_ast(extracted) == []

    # Catch common hallucinated/unsafe mistakes
    assert validate_cadquery_ast("")
    assert any("SyntaxError" in e for e in validate_cadquery_ast("def bad(:"))
    assert any("Disallowed import" in e for e in validate_cadquery_ast("import os\nimport cadquery as cq\nresult = 1"))
    assert any("show_object" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)\nshow_object(result)"))
    assert any("cutThroughAll" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(5,5,5).faces('>Z').circle(1).cutThroughAll()"))
    assert any("through" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(5,5,5).faces('>Z').hole(2, through=True)"))
    assert any("Cylinder" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Cylinder(2, 5)"))
    assert any("separate float arguments" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').moveTo((0, 0)).lineTo((10, 0)).close().extrude(5)"))
    assert any(".rLineTo()" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').moveTo(0, 0).rLineTo(10, 0).close().extrude(5)"))
    assert any(".rotateZ()" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3).rotateZ(45)"))
    assert any("ONE 3-tuple" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3).translate(1, 2, 3)"))
    assert any("Matrix" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3)\nm = cq.Matrix().rotate(1, 2, 3)"))
    assert any("Matrix" in e for e in validate_cadquery_ast("import cadquery as cq\nfrom cadquery import Matrix\nresult = Matrix()"))
    assert any("_rotate" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\nx = m._rotate((0,0,1), 45)"))
    assert any("corner=" in e for e in validate_cadquery_ast("import cadquery as cq\nresult = cq.Workplane('XY').box(1, 2, 3, corner=(0, 0, 0))"))
    assert any("Missing assignment to `result`" in e for e in validate_cadquery_ast("import cadquery as cq\npart = cq.Workplane('XY').box(1, 1, 1)"))


def test_executor_valid_multisolid_and_fillet_strip(tmp_path):
    valid_code = """
import cadquery as cq
result = cq.Workplane("XY").box(30, 20, 10, centered=(False, False, False))
"""
    res_ok = execute_cadquery_script(valid_code, tmp_path / "ok.stp")
    assert res_ok.ok and not res_ok.used_fillet_fallback
    assert (tmp_path / "ok.stp").is_file()
    tel = inspect_step_file(tmp_path / "ok.stp")
    assert tel.is_valid and tel.solids == 1
    assert tel.bounding_box_mm == {"x": 30.0, "y": 20.0, "z": 10.0}

    solid_union_code = """
import cadquery as cq
a = cq.Solid.makeBox(10, 10, 10)
b = cq.Solid.makeBox(10, 10, 10, cq.Vector(5, 0, 0))
result = a.union(b)
"""
    res_su = execute_cadquery_script(solid_union_code, tmp_path / "su.stp")
    assert not res_su.ok
    assert ".fuse(other)" in res_su.error

    solid_vals_code = """
import cadquery as cq
a = cq.Solid.makeBox(10, 10, 10)
result = a.vals()[0]
"""
    res_sv = execute_cadquery_script(solid_vals_code, tmp_path / "sv.stp")
    assert not res_sv.ok
    assert ".Solids()" in res_sv.error

    empty_cut_code = """
import cadquery as cq
base = cq.Workplane("XY").box(10, 10, 10)
result = cq.Workplane(cq.Plane(origin=(0, 0, 0), normal=(0, 0, 1))).circle(2).cutThruAll()
"""
    res_ec = execute_cadquery_script(empty_cut_code, tmp_path / "ec.stp")
    assert not res_ec.ok
    assert "extrude(L * 0.5, both=True)" in res_ec.error

    empty_union_code = """
import cadquery as cq
base = cq.Workplane("XY").box(10, 10, 10)
unextruded_sketch = cq.Workplane("XY").circle(3)
result = base.union(unextruded_sketch)
"""
    res_eu = execute_cadquery_script(empty_union_code, tmp_path / "eu.stp")
    assert not res_eu.ok
    assert "extruded/revolved into a 3D solid" in res_eu.error

    multi_code = """
import cadquery as cq
a = cq.Solid.makeBox(10, 10, 10)
b = cq.Solid.makeBox(10, 10, 10, cq.Vector(30, 0, 0))
result = cq.Compound.makeCompound([a, b])
"""
    res_multi = execute_cadquery_script(multi_code, tmp_path / "multi.stp")
    assert not res_multi.ok
    assert "Expected exactly 1 Solid, produced 2 solids" in res_multi.error
    assert "vol=1000.0" in res_multi.error and "X[30.0..40.0]" in res_multi.error

    # Verify strip_fillets_and_chamfers unwinds chained selectors cleanly
    bad_fillet_code = """
import cadquery as cq
result = (
    cq.Workplane("XY")
    .box(20, 20, 6, centered=(False, False, False))
    .edges("|Z")
    .fillet(15.0)
)
"""
    stripped = strip_fillets_and_chamfers(bad_fillet_code)
    assert stripped is not None
    assert "fillet" not in stripped and ".edges(" not in stripped

    res_salvaged = execute_cadquery_script(bad_fillet_code, tmp_path / "salvaged.stp")
    assert res_salvaged.ok
    assert res_salvaged.used_fillet_fallback
    tel_s = inspect_step_file(tmp_path / "salvaged.stp")
    assert tel_s.is_valid and tel_s.solids == 1 and abs(tel_s.volume_mm3 - 2400.0) < 1.0

    slow_code = """
import cadquery as cq
import time
time.sleep(3)
result = cq.Workplane("XY").box(5, 5, 5)
"""
    res_timeout = execute_cadquery_script(slow_code, tmp_path / "slow.stp", timeout_s=0.4)
    assert not res_timeout.ok
    assert "timed out" in res_timeout.error.lower()


def test_inspector_parametric_telemetry_and_example_outputs():
    # Parametric test solid: [0..80, 0..50, 0..12] plate with 2 Z through holes (d=10), 1 blind pocket (d=14, depth=5),
    # 1 cylindrical boss (d=12, h=6), and a sphere pole fillet check
    wp = (
        cq.Workplane("XY")
        .box(80.0, 50.0, 12.0, centered=(False, False, False))
        .faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .pushPoints([(-20.0, 0.0), (20.0, 0.0)])
        .hole(10.0)
        .faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(0.0, -12.0)
        .circle(7.0)
        .cutBlind(-5.0)
        .faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(0.0, 12.0)
        .circle(6.0)
        .extrude(6.0)
    )
    tel = inspect_shape(wp.val().wrapped)
    assert tel.is_valid
    assert tel.solids == 1
    assert tel.through_holes_genus == 2
    assert tel.bounding_box_mm == {"x": 80.0, "y": 50.0, "z": 18.0}

    expected_vol = (
        80 * 50 * 12
        - 2 * (math.pi * 5**2 * 12)
        - (math.pi * 7**2 * 5)
        + (math.pi * 6**2 * 6)
    )
    assert abs(tel.volume_mm3 - expected_vol) < 5.0
    assert len(tel.holes) == 3  # 2 through (d=10) + 1 blind (d=14)
    assert len(tel.bosses) == 1  # 1 cylindrical boss (d=12)

    facts_match = {
        "bounding_box_mm": {"x": 80.0, "y": 50.0, "z": 18.0},
        "volume_mm3": expected_vol,
        "surface_area_mm2": tel.surface_area_mm2,
        "through_holes_genus": 2,
        "cylindrical_holes_and_bores": [
            {"count": 2, "diameter": 10.0, "axis": "Z", "centres": [[20, 25, 6], [60, 25, 6]]},
        ],
        "cylindrical_shafts_and_bosses": [
            {"count": 1, "diameter": 12.0, "axis": "Z", "centres": [[40, 37, 15]]},
        ],
    }
    diff_ok = compare_against_facts(tel, facts_match)
    assert diff_ok.max_bbox_err_pct < 0.1
    assert abs(diff_ok.vol_err_pct) < 0.1
    assert diff_ok.genus_diff == 0
    assert diff_ok.missing_hole_groups == []
    assert diff_ok.missing_boss_groups == []
    assert not diff_ok.axis_swapped_hint

    # Shifted cylindrical boss/hole centres report radial_offset mismatch
    facts_shifted = dict(
        facts_match,
        cylindrical_shafts_and_bosses=[
            {"count": 1, "diameter": 12.0, "axis": "Z", "centres": [[40, 10, 15]]},
        ],
    )
    diff_shifted = compare_against_facts(tel, facts_shifted)
    assert diff_shifted.missing_boss_groups
    assert "radial_offset=27.0mm" in diff_shifted.missing_boss_groups[0]

    # Transposed axes triggers axis_swapped_hint
    facts_swapped = dict(facts_match, bounding_box_mm={"x": 80.0, "y": 18.0, "z": 50.0})
    diff_swap = compare_against_facts(tel, facts_swapped)
    assert diff_swap.axis_swapped_hint

    # Ensure example STEP files parse cleanly with inspect_step_file
    for p in sorted((REPO_ROOT / "artifacts/example_outputs").glob("test_case_*.stp")):
        ex_tel = inspect_step_file(p)
        assert ex_tel.is_valid and ex_tel.solids == 1 and ex_tel.volume_mm3 > 0


def test_renderer_prompts_diff_composite_and_flip_detection(tmp_path):
    # Asymmetric L-bracket along X, Y, Z with one through-hole
    l_shape = (
        cq.Workplane("XY")
        .box(60.0, 40.0, 12.0, centered=(False, False, False))
        .union(cq.Workplane("XY").box(18.0, 40.0, 45.0, centered=(False, False, False)))
        .faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .center(20.0, 8.0)
        .hole(8.0)
    )
    solid = l_shape.val().wrapped
    tel = inspect_shape(solid)
    bbox = tel.bounding_box_mm
    diag = (bbox["x"] ** 2 + bbox["y"] ** 2 + bbox["z"] ** 2) ** 0.5
    mesh = tessellate_shape(solid, diag_mm=diag)

    view_paths: list[Path] = []
    for name in VIEW_CAMERAS:
        rv = render_view(mesh, name, size=256)
        p = tmp_path / f"{name}.png"
        rv.image.save(p)
        view_paths.append(p)

    # Contact sheet packing + prefix-cache stability check
    sheets = build_contact_sheets(view_paths, tile_px=256)
    assert len(sheets) == 2
    for s_bytes in sheets:
        im = Image.open(io.BytesIO(s_bytes))
        assert im.size == (512, 512)

    u0 = build_initial_user_content("l_bracket", "Asymmetric bracket", None, view_paths, None, tile_px=256)
    repair_msgs = build_repair_messages(u0, "result = ...", "Missing hole")
    assert repair_msgs[0]["content"] == SYSTEM_PROMPT
    assert repair_msgs[1]["content"] == u0

    # Same bracket evaluates high Silhouette IoU, no handedness flip
    ve_same = evaluate_views(mesh, view_paths, eval_size=192)
    assert ve_same.mean_silhouette_iou > 0.90
    assert not ve_same.handedness_flip_detected

    diff_png = build_visual_diff_composite(ve_same, worst_n=2, tile_px=192)
    assert diff_png is not None
    diff_im = Image.open(io.BytesIO(diff_png))
    assert diff_im.size[0] == 3 * 192

    # Missing blind pocket (same outer silhouette) must render orange missing-internal-edge pixels
    import numpy as np

    pocket_target = (
        cq.Workplane("XY")
        .box(40.0, 30.0, 10.0, centered=(False, False, False))
        .faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .circle(8.0)
        .cutBlind(-4.0)
    )
    pt_mesh = tessellate_shape(pocket_target.val().wrapped, diag_mm=60.0)
    pt_top = tmp_path / "view_4_top.png"
    render_view(pt_mesh, "view_4_top", size=256).image.save(pt_top)
    solid_box_mesh = tessellate_shape(
        cq.Workplane("XY").box(40.0, 30.0, 10.0, centered=(False, False, False)).val().wrapped,
        diag_mm=60.0,
    )
    ve_pocket = evaluate_views(solid_box_mesh, [pt_top], eval_size=192)
    pocket_diff_png = build_visual_diff_composite(ve_pocket, worst_n=1, tile_px=192)
    assert pocket_diff_png is not None
    p_arr = np.asarray(Image.open(io.BytesIO(pocket_diff_png)).convert("RGB"))
    orange_px = np.all(p_arr == np.array([255, 165, 0], dtype=np.uint8), axis=-1)
    assert int(orange_px.sum()) > 10
    repair_with_img = build_repair_messages(u0, "result = ...", "Missing pocket", diff_composite_png=pocket_diff_png)
    assert "ORANGE" in repair_with_img[-1]["content"][1]["text"]

    # Dark shaded bronze faces on real target PNGs must not flood the crease-edge extractor
    tc5_iso = REPO_ROOT / "artifacts/inputs/test_case_5/images/view_6_iso_front_right_top.png"
    tc5_im = Image.open(tc5_iso).convert("RGB")
    tc5_arr = np.asarray(tc5_im)
    tc5_fg = np.any(tc5_arr < 245, axis=-1)
    tc5_edge = _extract_target_edge_mask(tc5_im, tc5_arr)
    assert 0.01 < (int(tc5_edge.sum()) / max(1, int(tc5_fg.sum()))) < 0.12

    # Mirror L-bracket across X axis -> evaluate against original PNGs must detect handedness flip
    l_mirrored = (
        cq.Workplane("XY")
        .box(60.0, 40.0, 12.0, centered=(False, False, False))
        .union(cq.Workplane("XY", origin=(42.0, 0.0, 0.0)).box(18.0, 40.0, 45.0, centered=(False, False, False)))
    )
    m_shape = l_mirrored.val().wrapped
    m_tel = inspect_shape(m_shape)
    m_mesh = tessellate_shape(m_shape, diag_mm=diag)
    ve_flip = evaluate_views(m_mesh, view_paths, eval_size=192)
    assert ve_flip.handedness_flip_detected
    m_diff = compare_against_facts(m_tel, None)
    sc_same = score_candidate(tel, None, m_diff, ve_same)
    sc_flip = score_candidate(m_tel, None, m_diff, ve_flip)
    assert sc_same.total > sc_flip.total


def test_fallback_step_on_tc1(tmp_path):
    tc1_dir = REPO_ROOT / "artifacts/inputs/test_case_1"
    views = sorted((tc1_dir / "images").glob("view_*.png"))
    facts = json.loads((tc1_dir / "facts.json").read_text(encoding="utf-8"))
    out_stp = tmp_path / "fb1.stp"
    ok = generate_fallback_step(views, facts, out_stp)
    assert ok
    tel = inspect_step_file(out_stp)
    assert tel.is_valid and tel.solids == 1 and tel.volume_mm3 > 100.0


def _make_rendered_target_part(tmp_path: Path):
    """Create target part + synthetic facts + 8 rendered PNGs for deterministic offline agent tests."""
    part = (
        cq.Workplane("XY")
        .box(40.0, 30.0, 10.0, centered=(False, False, False))
        .faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .hole(12.0)
    )
    shape = part.val().wrapped
    tel = inspect_shape(shape)
    diag = (40.0**2 + 30.0**2 + 10.0**2) ** 0.5
    mesh = tessellate_shape(shape, diag_mm=diag)
    views: list[Path] = []
    for name in VIEW_CAMERAS:
        p = tmp_path / f"{name}.png"
        render_view(mesh, name, size=256).image.save(p)
        views.append(p)
    facts = {
        "solids": 1,
        "faces": tel.faces,
        "edges": tel.edges,
        "through_holes_genus": tel.through_holes_genus,
        "bounding_box_mm": tel.bounding_box_mm,
        "volume_mm3": tel.volume_mm3,
        "surface_area_mm2": tel.surface_area_mm2,
        "cylindrical_holes_and_bores": [
            {"count": 1, "diameter": 12.0, "axis": "Z", "lengths": [10.0], "centres": [[20.0, 15.0, 5.0]]}
        ],
    }
    return views, facts


def test_agent_early_exit_multi_turn_repair_revert_and_fallback(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    views, facts = _make_rendered_target_part(target_dir)

    good_script = """```python
import cadquery as cq
result = (
    cq.Workplane("XY")
    .box(40.0, 30.0, 10.0, centered=(False, False, False))
    .faces(">Z")
    .workplane(centerOption="CenterOfBoundBox")
    .hole(12.0)
)
```"""
    no_hole_script = """```python
import cadquery as cq
result = cq.Workplane("XY").box(40.0, 30.0, 10.0, centered=(False, False, False))
```"""
    bad_syntax_script = """```python
import cadquery as cq
result = cq.Cylinder(5, 10)
```"""
    tiny_bad_script = """```python
import cadquery as cq
result = cq.Workplane("XY").box(5.0, 5.0, 5.0, centered=(False, False, False))
```"""

    # 1. Early exit on Turn 0
    cfg = _test_config(iou_early_exit=0.90)
    client_early = _FakeOpenAIClient(lambda idx, kw: good_script)
    agent1 = QwenCadAgent(config=cfg, client=client_early)
    s1 = agent1.generate_cad("Plate with hole", views, None, tmp_path / "early.stp", facts=facts, case_name="case_early")
    assert s1.turns_used == 1
    assert not s1.used_fallback
    assert s1.score > 0.90
    assert inspect_step_file(tmp_path / "early.stp").through_holes_genus == 1

    # 2. Multi-turn repair: Turn 0 hallucinated AST -> Turn 1 missing hole -> Turn 2 converged
    seq_replies = [bad_syntax_script, no_hole_script, good_script]
    client_repair = _FakeOpenAIClient(lambda idx, kw: seq_replies[idx])
    agent2 = QwenCadAgent(config=cfg, client=client_repair)
    s2 = agent2.generate_cad("Plate with hole", views, None, tmp_path / "repaired.stp", facts=facts, case_name="case_repair")
    assert s2.turns_used == 3
    assert not s2.used_fallback
    assert s2.score > 0.90
    # Turn 2 received visual diff composite + diagnostic feedback
    turn2_last_msg = client_repair.completions.calls[2]["messages"][-1]["content"]
    assert any(part.get("type") == "image_url" for part in turn2_last_msg)
    assert any("genus" in part.get("text", "").lower() for part in turn2_last_msg)

    # 3. Regression revert: Turn 0 decent -> Turn 1 tiny box -> Turn 2 verifies revert note, Turn 0 preserved
    cfg3 = _test_config(iou_early_exit=0.999, max_turns_simple=3)
    seq_revert = [no_hole_script, tiny_bad_script, no_hole_script]
    client_revert = _FakeOpenAIClient(lambda idx, kw: seq_revert[idx])
    agent3 = QwenCadAgent(config=cfg3, client=client_revert)
    s3 = agent3.generate_cad("Plate with hole", views, None, tmp_path / "reverted.stp", facts=facts, case_name="case_revert")
    assert s3.turns_used == 3
    turn2_txt = client_revert.completions.calls[2]["messages"][-1]["content"][0]["text"]
    assert "Reverting to the previous best script" in turn2_txt
    assert inspect_step_file(tmp_path / "reverted.stp").bounding_box_mm == {"x": 40.0, "y": 30.0, "z": 10.0}

    # 4. LLM failure triggers CPU fallback
    def _boom(idx, kw):
        raise ConnectionError("vLLM endpoint offline")

    agent4 = QwenCadAgent(config=cfg, client=_FakeOpenAIClient(_boom))
    s4 = agent4.generate_cad("Plate with hole", views, None, tmp_path / "fallback.stp", facts=facts, case_name="case_fb")
    assert s4.used_fallback
    tel_fb = inspect_step_file(tmp_path / "fallback.stp")
    assert tel_fb.is_valid and tel_fb.solids == 1

    # 5. Thinking-budget starvation retry: call 0 returns finish_reason="length" empty -> retries with enable_thinking=False
    class _ThinkingExhaustClient:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(content="", reasoning_content="thinking cut off..."),
                        )
                    ]
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=good_script, reasoning_content=None),
                    )
                ]
            )

    t_client = _ThinkingExhaustClient()
    agent5 = QwenCadAgent(config=cfg, client=t_client)
    s5 = agent5.generate_cad("Plate with hole", views, None, tmp_path / "think_retry.stp", facts=facts, case_name="case_tr")
    assert not s5.used_fallback
    assert len(t_client.calls) == 2
    assert t_client.calls[1].get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False

    # 6. Exact vLLM 400 BadRequestError ("Unexpected reasoning effort ...") recovery + valid budget values
    from openai import BadRequestError

    class _FakeBadRequest(BadRequestError):
        def __init__(self, msg: str):
            Exception.__init__(self, msg)

    class _EffortRejectClient:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("reasoning_effort") == "high":
                raise _FakeBadRequest(
                    "Error code: 400 - Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=good_script, reasoning_content=None),
                    )
                ]
            )

    e_client = _EffortRejectClient()
    agent6 = QwenCadAgent(config=cfg, client=e_client)
    assert agent6._plan_budget({"faces": 20})[0] == "medium"
    assert agent6._plan_budget({"faces": 150})[0] == "low"
    assert agent6._plan_budget({"faces": 350})[0] == "low"
    agent6_med = QwenCadAgent(config=_test_config(reasoning_effort="medium"), client=e_client)
    assert agent6_med._plan_budget({"faces": 350})[0] == "medium"
    reply6 = agent6._call_llm([{"role": "user", "content": "hi"}], reasoning_effort="high", timeout_s=30.0)
    assert "box(40.0, 30.0, 10.0" in reply6
    assert e_client.calls[1].get("reasoning_effort") == "medium"
    # Verify max_tokens caps for low vs none
    agent6._call_llm([{"role": "user", "content": "hi"}], reasoning_effort="low", timeout_s=30.0)
    assert e_client.calls[-1]["max_tokens"] <= 10240
    agent6._call_llm([{"role": "user", "content": "hi"}], reasoning_effort="none", timeout_s=30.0)
    assert e_client.calls[-1]["max_tokens"] <= 8192

    # 7a. Timeout rescue on complex part (starts "low") -> retries "none" and steps down Turn 1 to "none" (enable_thinking=False)
    from openai import APITimeoutError

    class _FakeTimeout(APITimeoutError):
        def __init__(self, msg: str):
            Exception.__init__(self, msg)

    class _TimeoutThenRepairClient:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise _FakeTimeout("Request timed out.")
            reply_txt = no_hole_script if len(self.calls) == 2 else good_script
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=reply_txt, reasoning_content=None),
                    )
                ]
            )

    to_client = _TimeoutThenRepairClient()
    complex_facts = dict(facts, faces=350)
    agent7 = QwenCadAgent(config=cfg, client=to_client)
    out7 = tmp_path / "timeout_rescue.stp"
    s7 = agent7.generate_cad("Complex plate", views, None, out7, facts=complex_facts, case_name="case_to")
    assert not s7.used_fallback
    assert s7.turns_used == 2
    assert to_client.calls[0].get("reasoning_effort") == "low"
    assert "reasoning_effort" not in to_client.calls[1]
    assert to_client.calls[1].get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert "reasoning_effort" not in to_client.calls[2]
    assert to_client.calls[2].get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert inspect_step_file(out7).through_holes_genus == 1
    assert not list(tmp_path.glob(".tmp_*"))

    # 7b. Timeout rescue on simple part (starts "medium") -> retries "none" and steps down Turn 1 to "low"
    to_client_med = _TimeoutThenRepairClient()
    simple_facts = dict(facts, faces=15)
    agent7b = QwenCadAgent(config=cfg, client=to_client_med)
    out7b = tmp_path / "timeout_rescue_med.stp"
    s7b = agent7b.generate_cad("Simple plate", views, None, out7b, facts=simple_facts, case_name="case_to_med")
    assert not s7b.used_fallback
    assert s7b.turns_used == 2
    assert to_client_med.calls[0].get("reasoning_effort") == "medium"
    assert to_client_med.calls[1].get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert to_client_med.calls[2].get("reasoning_effort") == "low"

    # 8. Stagnation plateau warning & early exit before burning all turns + Turn 2 drops effort to low
    cfg8 = _test_config(iou_early_exit=0.999, max_turns_simple=5)
    client8 = _FakeOpenAIClient(lambda idx, kw: no_hole_script)
    agent8 = QwenCadAgent(config=cfg8, client=client8)
    s8 = agent8.generate_cad("Plate with hole", views, None, tmp_path / "stagnant.stp", facts=facts, case_name="case_stag")
    assert s8.turns_used == 3  # stopped early at turn=2 instead of burning 5 turns
    assert client8.completions.calls[0].get("reasoning_effort") == "medium"
    assert client8.completions.calls[1].get("reasoning_effort") == "medium"
    assert client8.completions.calls[2].get("reasoning_effort") == "low"
    turn2_diag_txt = client8.completions.calls[2]["messages"][-1]["content"][0]["text"]
    assert "STAGNATION WARNING" in turn2_diag_txt


def test_submission_solid_check_rejects_mock_text(tmp_path):
    fake_stp = tmp_path / "test_case_1.stp"
    fake_stp.write_text("this is mock step content", encoding="utf-8")
    with pytest.raises(ValueError, match="STEP File could not be loaded"):
        cq.importers.importStep(str(fake_stp))


def test_inspector_collinear_cylinder_merge_and_separation():
    from agentic_cad.inspector import CylinderFeature, _merge_collinear_cylinders

    # Four 90-degree seam-split segments of one single bore -> merge into 1 full 360-deg feature
    quad_segs = [
        CylinderFeature(kind="hole", diameter=10.0, axis=(0.0, 0.0, 1.0), centre=(25.0, 25.0, 5.0), length=10.0, span_deg=90.0)
        for _ in range(4)
    ]
    merged_quad = _merge_collinear_cylinders(quad_segs)
    assert len(merged_quad) == 1
    assert merged_quad[0].span_deg == 360.0

    # Two separated collinear bores along Z (at Z=5 and Z=35, lengths=8) must stay distinct
    sep_bores = [
        CylinderFeature(kind="hole", diameter=10.0, axis=(0.0, 0.0, 1.0), centre=(25.0, 25.0, 5.0), length=8.0, span_deg=180.0),
        CylinderFeature(kind="hole", diameter=10.0, axis=(0.0, 0.0, 1.0), centre=(25.0, 25.0, 5.0), length=8.0, span_deg=180.0),
        CylinderFeature(kind="hole", diameter=10.0, axis=(0.0, 0.0, 1.0), centre=(25.0, 25.0, 35.0), length=8.0, span_deg=180.0),
        CylinderFeature(kind="hole", diameter=10.0, axis=(0.0, 0.0, 1.0), centre=(25.0, 25.0, 35.0), length=8.0, span_deg=180.0),
    ]
    merged_sep = _merge_collinear_cylinders(sep_bores)
    assert len(merged_sep) == 2
    assert all(m.span_deg == 360.0 for m in merged_sep)


def test_fallback_revolve_candidate_stepped_shaft_cone(tmp_path):
    # Axisymmetric stepped shaft + cone aligned along Z inside [0..40, 0..40, 0..60]
    plane = cq.Plane(origin=(20.0, 20.0, 0.0), xDir=(1.0, 0.0, 0.0), normal=(0.0, -1.0, 0.0))
    pts = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (10.0, 35.0), (10.0, 60.0), (0.0, 60.0)]
    solid = cq.Workplane(plane).polyline(pts).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0)).val()
    tel_tgt = inspect_shape(solid.wrapped)
    diag = (40.0**2 + 40.0**2 + 60.0**2) ** 0.5
    mesh = tessellate_shape(solid.wrapped, diag_mm=diag)
    views: list[Path] = []
    for name in VIEW_CAMERAS:
        p = tmp_path / f"{name}.png"
        render_view(mesh, name, size=256).image.save(p)
        views.append(p)
    facts = {
        "solids": 1,
        "faces": tel_tgt.faces,
        "edges": tel_tgt.edges,
        "through_holes_genus": 0,
        "bounding_box_mm": tel_tgt.bounding_box_mm,
        "volume_mm3": tel_tgt.volume_mm3,
        "surface_area_mm2": tel_tgt.surface_area_mm2,
    }
    out_stp = tmp_path / "rev_fb.stp"
    assert generate_fallback_step(views, facts, out_stp)
    tel_fb = inspect_step_file(out_stp)
    assert tel_fb.is_valid and tel_fb.solids == 1 and tel_fb.shells == 1
    assert tel_fb.face_types.get("CONE", 0) > 0 or tel_fb.face_types.get("CYLINDER", 0) > 0
    sh_fb = load_step_shape(out_stp)
    ve = evaluate_views(tessellate_shape(sh_fb, diag_mm=diag), views, eval_size=192)
    assert ve.mean_silhouette_iou > 0.90


def test_prompts_ortho_envelope_profiles(tmp_path):
    from agentic_cad.prompts import _extract_ortho_envelope_profiles

    # facts=None -> no ortho envelope section appended
    tc5_dir = REPO_ROOT / "artifacts/inputs/test_case_5"
    tc5_views = sorted((tc5_dir / "images").glob("view_*.png"))
    tc5_facts = json.loads((tc5_dir / "facts.json").read_text(encoding="utf-8"))
    u_nofacts = build_initial_user_content("p", "desc", None, tc5_views, None, tile_px=256)
    assert "Orthographic Silhouette Outer Envelopes" not in u_nofacts[0]["text"]

    # On test_case_5, Right YZ reveals main body Y[22.3..85.7] vs top projecting tabs Y[0.0..85.7]
    u_tc5 = build_initial_user_content("tc5", "desc", tc5_facts, tc5_views, None, tile_px=256)
    txt = u_tc5[0]["text"]
    assert "Orthographic Silhouette Outer Envelopes" in txt
    assert "Front XZ:" in txt and "Right YZ:" in txt and "Top XY:" in txt
    assert "Y[22." in txt and "Y[0.0..85." in txt

    # Synthetic L-shape test using fallback view_5_bottom (tests vertical flip in VIEW_CAMERAS)
    l_shape = (
        cq.Workplane("XY")
        .box(60.0, 40.0, 10.0, centered=(False, False, False))
        .union(cq.Workplane("XY").box(60.0, 15.0, 10.0, centered=(False, False, False)).translate((0.0, 25.0, 10.0)))
    )
    solid = l_shape.val().wrapped
    tel = inspect_shape(solid)
    mesh = tessellate_shape(solid, diag_mm=80.0)
    bot_path = tmp_path / "view_5_bottom.png"
    front_path = tmp_path / "view_0_front.png"
    render_view(mesh, "view_5_bottom", size=256).image.save(bot_path)
    render_view(mesh, "view_0_front", size=256).image.save(front_path)
    # Narrow tab sits at Y=[25..40], i.e., large Y, while bottom base spans X=[0..60]
    syn_facts = {"bounding_box_mm": tel.bounding_box_mm}
    prof = _extract_ortho_envelope_profiles([front_path, bot_path], syn_facts)
    assert "Top XY:" in prof
    assert "Y=2.0->X[0.0..60.0]" in prof


def test_cli_main_worker_execution_and_strict_output_filenames(tmp_path, monkeypatch):
    # Prepare 2 synthetic test cases laid out exactly like artifacts/inputs
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    for idx in (1, 2):
        case_dir = in_dir / f"test_case_{idx}"
        img_dir = case_dir / "images"
        img_dir.mkdir(parents=True)
        views, facts = _make_rendered_target_part(img_dir)
        (case_dir / "description.md").write_text(f"Test bracket {idx}", encoding="utf-8")
        (case_dir / "facts.json").write_text(json.dumps(facts), encoding="utf-8")

    monkeypatch.setenv("CAD_MAX_WORKERS", "2")
    monkeypatch.setenv("CAD_IOU_EARLY_EXIT", "0.90")
    monkeypatch.setattr(
        QwenCadAgent,
        "_call_llm",
        lambda self, messages, reasoning_effort, timeout_s: """```python
import cadquery as cq
result = (
    cq.Workplane("XY")
    .box(40.0, 30.0, 10.0, centered=(False, False, False))
    .faces(">Z")
    .workplane(centerOption="CenterOfBoundBox")
    .hole(12.0)
)
```""",
    )

    cli_main.main(argparse.Namespace(input_dir=in_dir, output_dir=out_dir))
    written = sorted(p.name for p in out_dir.iterdir())
    assert written == ["test_case_1.stp", "test_case_2.stp"]
    for p in out_dir.iterdir():
        tel = inspect_step_file(p)
        assert tel.is_valid and tel.solids == 1

    # Also verify inspect_outputs script writes diff PNGs and STL meshes cleanly
    import scripts.inspect_outputs as inspect_cli

    render_dir = tmp_path / "renders"
    stl_dir = tmp_path / "stls"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_outputs.py",
            "--input_dir",
            str(in_dir),
            "--output_dir",
            str(out_dir),
            "--render_dir",
            str(render_dir),
            "--stl_dir",
            str(stl_dir),
            "--eval_px",
            "192",
            "--cases",
            "1",
        ],
    )
    inspect_cli.main()
    assert (render_dir / "test_case_1_diff.png").is_file()
    assert (stl_dir / "test_case_1.stl").stat().st_size > 100
    assert not (stl_dir / "test_case_2.stl").exists()
