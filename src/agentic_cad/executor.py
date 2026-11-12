from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentic_cad.cadquery_ref import validate_cadquery_ast

_RUNNER_SCRIPT = r'''
import json
import sys
import traceback
from pathlib import Path

import cadquery as cq

script_path = Path(sys.argv[1])
step_out = Path(sys.argv[2])
meta_out = Path(sys.argv[3])

code = script_path.read_text(encoding="utf-8")
ns = {"__name__": "__main__"}
try:
    exec(compile(code, str(script_path), "exec"), ns, ns)
except Exception as exc:
    meta_out.write_text(json.dumps({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }))
    sys.exit(2)

res = ns.get("result")
if res is None:
    meta_out.write_text(json.dumps({
        "ok": False,
        "error": "Variable `result` is None or not defined.",
        "traceback": "",
    }))
    sys.exit(3)

try:
    if isinstance(res, cq.Workplane):
        solids = res.solids().vals()
        if not solids:
            # Check if stack contains a Compound/Solid directly
            vals = res.vals()
            solids = [v for s in vals if isinstance(s, cq.Shape) for v in s.Solids()]
        n_solids = len(solids)
        export_obj = res if n_solids == 1 else (cq.Workplane(obj=solids[0]) if n_solids > 0 else None)
    elif isinstance(res, cq.Shape):
        solids = res.Solids()
        n_solids = len(solids)
        export_obj = solids[0] if n_solids >= 1 else None
    else:
        meta_out.write_text(json.dumps({
            "ok": False,
            "error": f"`result` must be cadquery.Workplane or cadquery.Shape, got {type(res).__name__}.",
            "traceback": "",
        }))
        sys.exit(4)

    if n_solids != 1 or export_obj is None:
        meta_out.write_text(json.dumps({
            "ok": False,
            "error": f"Expected exactly 1 Solid, produced {n_solids} solids.",
            "traceback": "",
        }))
        sys.exit(5)

    cq.exporters.export(export_obj, str(step_out), "STEP")
    meta_out.write_text(json.dumps({"ok": True, "error": "", "traceback": ""}))
except Exception as exc:
    meta_out.write_text(json.dumps({
        "ok": False,
        "error": f"ExportError ({type(exc).__name__}): {exc}",
        "traceback": traceback.format_exc(),
    }))
    sys.exit(6)
'''


@dataclass
class ExecutionResult:
    ok: bool
    step_path: Path | None
    error: str
    traceback_text: str
    used_fillet_fallback: bool = False


class _StripFinishingTransformer(ast.NodeTransformer):
    """Strip chained .fillet(...) / .chamfer(...) and their preceding selectors cleanly."""

    _SELECTOR_ATTRS = {"edges", "faces", "vertices", "wires", "solids", "shells", "tag"}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"fillet", "chamfer"}:
            cur = node.func.value
            while (
                isinstance(cur, ast.Call)
                and isinstance(cur.func, ast.Attribute)
                and cur.func.attr in self._SELECTOR_ATTRS
            ):
                cur = cur.func.value
            return cur
        return node


def strip_fillets_and_chamfers(code: str) -> str | None:
    """Return a version of `code` with .fillet/.chamfer calls removed, or None if unchanged."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    new_tree = _StripFinishingTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    stripped = ast.unparse(new_tree)
    return stripped if stripped != ast.unparse(ast.parse(code)) else None


def _run_subprocess(code: str, step_out: Path, timeout_s: float) -> tuple[bool, str, str]:
    with tempfile.TemporaryDirectory(prefix="cad_exec_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        runner_py = tmp_path / "_run_cq.py"
        script_py = tmp_path / "candidate.py"
        meta_json = tmp_path / "meta.json"
        runner_py.write_text(_RUNNER_SCRIPT, encoding="utf-8")
        script_py.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("HOME", tmp_dir)
        env.setdefault("MPLCONFIGDIR", tmp_dir)

        try:
            proc = subprocess.run(
                [sys.executable, str(runner_py), str(script_py), str(step_out), str(meta_json)],
                cwd=tmp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"Script execution timed out after {timeout_s:.0f}s (likely heavy boolean/sweep).", ""

        if meta_json.is_file():
            try:
                data = json.loads(meta_json.read_text(encoding="utf-8"))
                return bool(data.get("ok")), str(data.get("error", "")), str(data.get("traceback", ""))
            except Exception:
                pass

        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = stderr or stdout or f"Subprocess crashed with exit code {proc.returncode}"
        return False, msg, stderr


def execute_cadquery_script(
    code: str,
    step_out: Path,
    timeout_s: float = 45.0,
    allow_fillet_strip_fallback: bool = True,
) -> ExecutionResult:
    """Validate AST and execute CadQuery script in an isolated subprocess writing STEP output."""
    ast_errors = validate_cadquery_ast(code)
    if ast_errors:
        return ExecutionResult(
            ok=False,
            step_path=None,
            error="Static AST check failed:\n" + "\n".join(f"- {e}" for e in ast_errors),
            traceback_text="",
        )

    step_out.parent.mkdir(parents=True, exist_ok=True)
    ok, err, tb = _run_subprocess(code, step_out, timeout_s=timeout_s)
    if ok and step_out.is_file() and step_out.stat().st_size > 0:
        return ExecutionResult(ok=True, step_path=step_out, error="", traceback_text="")

    # Automatic pre-fillet salvage if failure occurred during OCCT BRep finishing
    is_brep_or_timeout = (
        "StdFail" in err
        or "BRep_API" in err
        or "NotDone" in err
        or "fillet" in tb.lower()
        or "chamfer" in tb.lower()
        or "timed out" in err.lower()
    )
    if allow_fillet_strip_fallback and is_brep_or_timeout:
        stripped = strip_fillets_and_chamfers(code)
        if stripped:
            ok2, err2, tb2 = _run_subprocess(stripped, step_out, timeout_s=timeout_s)
            if ok2 and step_out.is_file() and step_out.stat().st_size > 0:
                return ExecutionResult(
                    ok=True,
                    step_path=step_out,
                    error=f"(Original fillet/chamfer failed: {err}; salvaged unfilleted solid)",
                    traceback_text=tb,
                    used_fillet_fallback=True,
                )

    if "object has no attribute 'union'" in err and any(k in err for k in ("'Solid'", "'Compound'", "'Shape'")):
        err += "\nHint: cq.Solid/cq.Shape objects do not have .union(); use .fuse(other) or wrap in cq.Workplane('XY').newObject([solid])."
    elif "object has no attribute 'faces'" in err and any(k in err for k in ("'Solid'", "'Compound'", "'Shape'")):
        err += "\nHint: cq.Solid/cq.Shape objects do not have .faces(); wrap in cq.Workplane('XY').newObject([solid]) before selecting faces."

    return ExecutionResult(ok=False, step_path=None, error=err, traceback_text=tb)
