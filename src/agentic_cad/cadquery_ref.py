from __future__ import annotations

import ast
import re

CADQUERY_CHEATSHEET = """## Authoritative CadQuery 2.8 Reference (Strict BRep Rules)

1. **Canonical Coordinate Frame (CRITICAL)**:
   - Build the part directly inside `[0..X, 0..Y, 0..Z]` where `(0, 0, 0)` is the bounding-box min corner.
   - Start rectangular parts with `cq.Workplane("XY").box(X, Y, Z, centered=(False, False, False))`.
   - Note Workplane axes & normals (right-handed):
     - `"XY"`: local `x=+X`, `y=+Y`, normal `+Z` (`extrude(+d)` goes `+Z`).
     - `"YZ"`: local `x=+Y`, `y=+Z`, normal `+X` (`extrude(+d)` goes `+X`).
     - `"XZ"`: local `x=+X`, `y=+Z`, normal **`-Y`**! (`extrude(+d)` goes **`-Y`**! To extrude toward `+Y` from `Y=y0`, keep `cq.Workplane("XZ", origin=(0, y0, 0))` and pass **negative depth** `.extrude(-d)` — OR place the plane at `origin=(0, y0 + d, 0)` and `.extrude(+d)`. Do NOT use `normal=(0, 1, 0)` with `xDir=(1, 0, 0)`, because the right-hand rule flips local `y` to `-Z`!)
   - Do NOT mirror handedness. Position & global orientation match the 8 views (`view_0_front` looks `+Y`, `view_4_top` looks `-Z`).

2. **Verified Method Signatures & Common Traps**:
   - Prefer `.circle(r).extrude(h)` or `cq.Solid.makeCylinder(radius, height, pnt, dir)` over `Workplane.cylinder(height, radius)` (note `height` is FIRST in `Workplane.cylinder`!).
   - **`cutBlind` direction**: `.faces(">Z").workplane(centerOption="CenterOfBoundBox")` has outward normal `+Z` -> pass **negative depth** `.cutBlind(-d)` (or `workplane(invert=True)` + `.cutBlind(d)`). Positive `.cutBlind(+d)` on an outward face cuts into empty air!
   - **Holes** (already invert inward automatically on selected planar faces):
     - `.hole(diameter, depth=None)` — omit `depth` (or pass `None`) for a through-hole. NEVER pass `through=True`.
     - `.cboreHole(diameter, cboreDiameter, cboreDepth, depth=None)`
     - `.cskHole(diameter, cskDiameter, cskAngle, depth=None)` (typical `cskAngle=82` or `90` degrees).
   - **Always pass `centerOption="CenterOfBoundBox"`** on `.faces(...).workplane(centerOption="CenterOfBoundBox")` unless setting an explicit `cq.Plane(origin=..., normal=...)`.
   - **Revolve**: `axisStart` and `axisEnd` are 2D points in the active Workplane's local `(x, y)` coordinates! E.g. on `Workplane("XZ")`, local `y` is global `+Z`, so `.revolve(360, (0, 0), (0, 1))` revolves around the global Z-axis passing through `(0, 0, 0)`. Keep profile strictly on one side (`x >= 0`), touching axis cleanly if solid.
   - **Batch repeated holes/cuts/bosses**: Use `.pushPoints([(x1, y1), ...])` or `.polarArray(radius, startAngle, angle, count)` or union/cut a `cq.Compound.makeCompound(shapes)` once instead of calling `.cut()` / `.union()` inside a large Python `for` loop.
   - **Slots & Polygons**: `.slot2D(length, diameter, angle=0)` (`length` is end-to-end), `.polyline(pts).close()`, `.threePointArc(mid_pt, end_pt)`, `.radiusArc(end_pt, radius)`.

3. **Selectors & Tagging**:
   - `">Z"` highest centroid Z; `"<Z"` lowest centroid Z; `"|Z"` faces/edges parallel to Z (BOTH top & bottom!); `"+Z"` outward normal `+Z`; `"#Z"` perpendicular to Z; `"%Plane"` / `"%Cylinder"` surface type.
   - Tag faces before topological changes shift indices: `.faces(">>Z[1]").tag("ledge").end()` then `.faces(tag="ledge").workplane(...)`.

4. **Resilient Fillets / Chamfers (Never Crash the Base Solid)**:
   - OpenCASCADE `BRep_API: command not done` happens when fillet radius exceeds edge length or overlaps adjacent fillets.
   - Group edge fillets/chamfers AFTER base geometry + holes, smallest radius first, and wrap risky fillet/chamfer steps cleanly:
     ```python
     try:
         result = result.edges("|Z").fillet(2.5)
     except Exception:
         pass
     ```

5. **Script Output Contract**:
   - Pure Python using `import cadquery as cq` (and standard `math` if needed).
   - Final single `Solid` (`Workplane` or `Shape`) MUST be assigned to variable `result`.
   - Do NOT call `show_object(...)`, `cq.exporters.export(...)`, `open(...)`, or import `os`/`pathlib`/`sys`.
"""

_DISALLOWED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "pathlib",
    "importlib",
    "ctypes",
    "multiprocessing",
}

_DISALLOWED_CALLS = {"eval", "exec", "__import__", "open", "compile", "show_object"}

_HALLUCINATED_METHODS = {
    "cutThroughAll": "Use `.cutThruAll()` instead of `.cutThroughAll()`.",
    "fillet_all": "CadQuery has no `.fillet_all()`; select edges via `.edges(...)` and call `.fillet(r)`.",
    "chamfer_all": "CadQuery has no `.chamfer_all()`; select edges via `.edges(...)` and call `.chamfer(d)`.",
}

_HALLUCINATED_CQ_ATTRS = {
    "Cylinder": "Use `cq.Workplane(...).circle(r).extrude(h)` or `cq.Solid.makeCylinder(r, h, pnt, dir)` instead of `cq.Cylinder`.",
    "Box": "Use `cq.Workplane('XY').box(x, y, z)` or `cq.Solid.makeBox(x, y, z)` instead of `cq.Box`.",
    "Cone": "Use `cq.Solid.makeCone(r1, r2, h, pnt, dir)` instead of `cq.Cone`.",
    "Sphere": "Use `cq.Workplane('XY').sphere(r)` or `cq.Solid.makeSphere(r)` instead of `cq.Sphere`.",
}


def extract_python_code(raw_text: str) -> str:
    """Strip <think> blocks and extract Python code from markdown fences or raw response."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    fenced_py = re.findall(r"```(?:python|py)\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_py:
        # Pick longest valid-looking block if multiple blocks exist
        return max(fenced_py, key=len).strip()
    fenced_any = re.findall(r"```\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fenced_any:
        return max(fenced_any, key=len).strip()
    return cleaned


def validate_cadquery_ast(code: str) -> list[str]:
    """Run static AST safety and anti-hallucination checks before touching OpenCASCADE."""
    errors: list[str] = []
    if not code.strip():
        return ["Generated script is empty."]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Python SyntaxError at line {exc.lineno}: {exc.msg}"]

    has_cad_import = False
    assigns_result = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _DISALLOWED_IMPORTS:
                    errors.append(f"Disallowed import `{alias.name}`; script must stay pure CadQuery/math.")
                if top in {"cadquery", "OCP"}:
                    has_cad_import = True
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in _DISALLOWED_IMPORTS:
                errors.append(f"Disallowed import from `{node.module}`; script must stay pure CadQuery/math.")
            if mod in {"cadquery", "OCP"}:
                has_cad_import = True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    assigns_result = True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "result":
                assigns_result = True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALLS:
                if node.func.id == "show_object":
                    errors.append("Do not call `show_object(...)`; assign the final solid to `result`.")
                else:
                    errors.append(f"Disallowed call `{node.func.id}(...)`.")
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in _HALLUCINATED_METHODS:
                    errors.append(_HALLUCINATED_METHODS[attr])
                if attr in {"hole", "cboreHole", "cskHole"}:
                    for kw in node.keywords:
                        if kw.arg == "through":
                            errors.append(
                                f"`.{attr}()` has no `through` argument; omit `depth` (or set `depth=None`) for through-holes."
                            )
                        elif kw.arg == "diam":
                            errors.append(f"`.{attr}()` uses `diameter=...`, not `diam=...`.")
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "cq"
                    and attr in _HALLUCINATED_CQ_ATTRS
                ):
                    errors.append(_HALLUCINATED_CQ_ATTRS[attr])
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "cq" and node.attr in _HALLUCINATED_CQ_ATTRS:
                msg = _HALLUCINATED_CQ_ATTRS[node.attr]
                if msg not in errors:
                    errors.append(msg)

    if not has_cad_import:
        errors.append("Missing `import cadquery as cq` (or `from cadquery.func import *`).")
    if not assigns_result:
        errors.append("Missing assignment to `result` (the final single-solid Workplane/Shape must be bound to `result`).")

    return errors
