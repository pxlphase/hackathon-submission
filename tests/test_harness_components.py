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

    multi_code = """
import cadquery as cq
a = cq.Solid.makeBox(10, 10, 10)
b = cq.Solid.makeBox(10, 10, 10, cq.Vector(30, 0, 0))
result = cq.Compound.makeCompound([a, b])
"""
    res_multi = execute_cadquery_script(multi_code, tmp_path / "multi.stp")
    assert not res_multi.ok
    assert "Expected exactly 1 Solid, produced 2 solids" in res_multi.error

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
    }
    diff_ok = compare_against_facts(tel, facts_match)
    assert diff_ok.max_bbox_err_pct < 0.1
    assert abs(diff_ok.vol_err_pct) < 0.1
    assert diff_ok.genus_diff == 0
    assert diff_ok.missing_hole_groups == []
    assert not diff_ok.axis_swapped_hint

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
    assert agent6._plan_budget({"faces": 150})[0] == "medium"
    assert agent6._plan_budget({"faces": 350})[0] == "xhigh"
    reply6 = agent6._call_llm([{"role": "user", "content": "hi"}], reasoning_effort="high", timeout_s=30.0)
    assert "box(40.0, 30.0, 10.0" in reply6
    assert e_client.calls[1].get("reasoning_effort") == "medium"

    # 7. Timeout rescue on Turn 0 -> retries fast low-reasoning pass + repair turns cap xhigh down to medium
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
    assert to_client.calls[0].get("reasoning_effort") == "xhigh"
    assert to_client.calls[1].get("reasoning_effort") == "low"
    assert to_client.calls[2].get("reasoning_effort") == "medium"
    assert inspect_step_file(out7).through_holes_genus == 1
    assert not list(tmp_path.glob(".tmp_*"))


def test_submission_solid_check_rejects_mock_text(tmp_path):
    fake_stp = tmp_path / "test_case_1.stp"
    fake_stp.write_text("this is mock step content", encoding="utf-8")
    with pytest.raises(ValueError, match="STEP File could not be loaded"):
        cq.importers.importStep(str(fake_stp))


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
