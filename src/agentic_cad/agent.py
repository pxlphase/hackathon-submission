from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APITimeoutError, BadRequestError, OpenAI

from agentic_cad.cadquery_ref import extract_python_code
from agentic_cad.config import HarnessConfig
from agentic_cad.executor import execute_cadquery_script
from agentic_cad.fallback import generate_fallback_step
from agentic_cad.inspector import CadTelemetry, FactsDiff, compare_against_facts, inspect_shape, load_step_shape
from agentic_cad.prompts import (
    SYSTEM_PROMPT,
    _png_bytes_to_data_url,
    build_initial_user_content,
    build_repair_messages,
)
from agentic_cad.renderer import ViewEvalResult, build_visual_diff_composite, evaluate_views, tessellate_shape
from agentic_cad.scorer import SurrogateScore, score_candidate
from agentic_cad.tools import (
    CAD_TOOL_SCHEMAS,
    measure_ortho_features,
    parse_text_tool_calls,
    run_cad_probe,
    zoom_view_crop,
)

logger = logging.getLogger(__name__)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` via a same-directory temp file + os.replace to avoid torn reads/writes."""
    tmp_dst = dst.parent / f".tmp_{dst.name}_{os.getpid()}_{threading.get_ident()}"
    try:
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)
    finally:
        if tmp_dst.exists():
            tmp_dst.unlink(missing_ok=True)


def _prefer_llm_over_floor(
    best_score: SurrogateScore | None,
    best_diff: FactsDiff | None,
    best_ve: ViewEvalResult | None,
    floor_score: SurrogateScore | None,
    floor_diff: FactsDiff | None,
) -> bool:
    """Decide whether the best verified LLM solid should supersede the deterministic visual-hull floor."""
    if best_score is None or not best_score.is_valid_single_solid:
        return False
    if floor_score is None or best_score.total >= floor_score.total:
        return True
    return (
        best_diff is not None
        and floor_diff is not None
        and best_ve is not None
        and best_diff.genus_diff == 0
        and best_diff.max_bbox_err_pct <= 2.5
        and best_ve.mean_silhouette_iou >= 0.80
        and abs(floor_diff.vol_err_pct) >= 8.0
        and abs(best_diff.vol_err_pct) < 0.5 * abs(floor_diff.vol_err_pct)
        and best_score.total >= floor_score.total - 0.10
    )


@dataclass
class GenerationSummary:
    case_name: str
    output_path: Path
    turns_used: int
    score: float
    mean_iou: float
    used_fallback: bool
    elapsed_s: float


@dataclass
class _BranchOutcome:
    ok: bool
    llm_failed: bool = False
    code: str = ""
    step_path: Path | None = None
    telemetry: CadTelemetry | None = None
    diff: FactsDiff | None = None
    view_eval: ViewEvalResult | None = None
    score: SurrogateScore | None = None
    diag_report: str = ""
    used_fillet_fallback: bool = False
    exec_note: str = ""
    stepdown_effort: str | None = None
    early_exit: bool = False


def _normalize_tool_args(raw_args: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw_args, dict):
        return raw_args, None
    if raw_args is None or raw_args == "":
        return {}, None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed, None
            return {}, f"Tool arguments must decode to a JSON object, got {type(parsed).__name__}."
        except json.JSONDecodeError as exc:
            return {}, f"Invalid JSON tool arguments: {exc}"
    return {}, f"Unsupported arguments type: {type(raw_args).__name__}"


class QwenCadAgent:
    """Autonomous multi-turn CAD synthesis and verification agent backed by Qwen3.8-27B."""

    def __init__(self, config: HarnessConfig | None = None, client: Any | None = None) -> None:
        self.config = config or HarnessConfig.from_env()
        self.client = client or OpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            timeout=self.config.llm_timeout_s,
            max_retries=0,
        )
        self._state_lock = threading.Lock()
        self._tools_supported = bool(self.config.enable_tools)

    def _plan_budget(self, facts: dict[str, Any] | None) -> tuple[str, int]:
        """Choose reasoning effort and max repair turns from BRep face complexity and config."""
        faces = int((facts or {}).get("faces") or 50)
        if faces <= 30:
            turns = self.config.max_turns_simple
            auto_effort = "medium"
        elif faces <= 200:
            turns = self.config.max_turns_medium
            auto_effort = "low"
        else:
            turns = self.config.max_turns_complex
            auto_effort = "low"
        cfg_eff = (self.config.reasoning_effort or "auto").strip().lower()
        effort = cfg_eff if cfg_eff in ("low", "medium", "none") else auto_effort
        return effort, turns

    def _is_call_llm_monkeypatched(self) -> bool:
        return getattr(self._call_llm, "__func__", None) is not _ORIGINAL_CALL_LLM

    def _invoke_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        reasoning_effort: str,
        timeout_s: float,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> tuple[str, list[dict[str, Any]], str, list[dict[str, Any]] | None]:
        t_mode = self.config.thinking_mode
        max_tok = self.config.max_tokens
        if len(messages) > 2:
            max_tok = min(max_tok, 12288)
        if reasoning_effort == "low":
            max_tok = min(max_tok, 10240)
        elif reasoning_effort == "none":
            max_tok = min(max_tok, 8192)
        kwargs: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed if seed is None else seed,
            "timeout": timeout_s,
        }
        if tools:
            kwargs["tools"] = tools
        if t_mode == "off" or reasoning_effort == "none":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            kwargs["reasoning_effort"] = reasoning_effort

        def _invoke(call_kwargs: dict[str, Any]):
            for _ in range(4):
                try:
                    return self.client.chat.completions.create(**call_kwargs)
                except BadRequestError as exc:
                    err_s = str(exc).lower()
                    if "tools" in call_kwargs and any(
                        tok in err_s for tok in ("tool", "function", "auto-tool-choice", "tool_call_parser")
                    ):
                        with self._state_lock:
                            self._tools_supported = False
                        call_kwargs.pop("tools", None)
                        continue
                    if "reasoning" in err_s and "reasoning_effort" in call_kwargs:
                        if call_kwargs["reasoning_effort"] != "medium" and "medium" in err_s:
                            call_kwargs["reasoning_effort"] = "medium"
                        else:
                            call_kwargs.pop("reasoning_effort", None)
                        continue
                    if "chat_template_kwargs" in err_s and "extra_body" in call_kwargs:
                        call_kwargs.pop("extra_body", None)
                        continue
                    raise
            return self.client.chat.completions.create(**call_kwargs)

        resp = _invoke(dict(kwargs))
        choice = resp.choices[0]
        msg = choice.message
        raw_content = (getattr(msg, "content", None) or "").strip()
        native_calls = getattr(msg, "tool_calls", None) or []

        parsed_calls: list[dict[str, Any]] = []
        raw_assistant_tool_calls: list[dict[str, Any]] | None = None
        if native_calls:
            raw_assistant_tool_calls = []
            for idx, tc in enumerate(native_calls):
                tc_id = getattr(tc, "id", None) or f"call_{idx}"
                fn = getattr(tc, "function", None)
                fn_name = getattr(fn, "name", "") if fn is not None else ""
                fn_args = getattr(fn, "arguments", "{}") if fn is not None else "{}"
                args_str = fn_args if isinstance(fn_args, str) else json.dumps(fn_args)
                raw_assistant_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": args_str},
                    }
                )
                parsed_calls.append({"id": tc_id, "name": fn_name, "arguments": fn_args, "source": "native"})
        else:
            cleaned_txt, text_calls = parse_text_tool_calls(raw_content)
            if text_calls:
                for tc in text_calls:
                    tc["source"] = "text"
                parsed_calls = text_calls
                content_for_code = cleaned_txt
            else:
                content_for_code = raw_content

        if native_calls:
            content_for_code = raw_content

        if not parsed_calls and not content_for_code:
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
            if "```" in reasoning:
                content_for_code = reasoning
                raw_content = reasoning

        # Guard against thinking-budget starvation returning empty/cutoff reply: retry once with thinking disabled
        finish_reason = getattr(choice, "finish_reason", None)
        if (
            not parsed_calls
            and t_mode == "auto"
            and reasoning_effort != "none"
            and (not content_for_code or (finish_reason == "length" and "```" not in content_for_code))
        ):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("reasoning_effort", None)
            retry_kwargs.pop("tools", None)
            retry_kwargs["max_tokens"] = min(int(retry_kwargs["max_tokens"]), 8192)
            retry_kwargs["timeout"] = min(150.0, timeout_s)
            retry_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            try:
                resp2 = _invoke(retry_kwargs)
                content2 = (getattr(resp2.choices[0].message, "content", None) or "").strip()
                if content2:
                    content_for_code = content2
                    raw_content = content2
            except Exception:
                pass
        return content_for_code, parsed_calls, raw_content, raw_assistant_tool_calls

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        reasoning_effort: str,
        timeout_s: float,
    ) -> str:
        content, _, _, _ = self._invoke_completion(
            messages=messages,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            tools=None,
        )
        return content

    def _evaluate_step(
        self,
        step_path: Path,
        facts: dict[str, Any] | None,
        views: list[Path],
    ) -> tuple[CadTelemetry, FactsDiff, ViewEvalResult, SurrogateScore]:
        shape = load_step_shape(step_path)
        telemetry = inspect_shape(shape)
        diff = compare_against_facts(telemetry, facts)
        bbox = telemetry.bounding_box_mm
        diag = max(10.0, (bbox["x"] ** 2 + bbox["y"] ** 2 + bbox["z"] ** 2) ** 0.5)
        mesh = tessellate_shape(shape, diag_mm=diag)
        view_eval = evaluate_views(mesh, views, eval_size=self.config.render_eval_px)
        score = score_candidate(telemetry, facts, diff, view_eval)
        return telemetry, diff, view_eval, score

    def _should_early_exit(
        self,
        diff: Any,
        view_eval: Any,
        score: SurrogateScore,
        facts: dict[str, Any] | None,
    ) -> bool:
        if not score.is_valid_single_solid:
            return False
        if view_eval.handedness_flip_detected:
            return False
        if view_eval.mean_silhouette_iou < self.config.iou_early_exit:
            return False
        if view_eval.min_silhouette_iou < min(0.88, self.config.iou_early_exit):
            return False
        if not facts:
            return True
        return (
            diff.max_bbox_err_pct <= self.config.bbox_tol_pct
            and abs(diff.vol_err_pct) <= self.config.vol_tol_pct
            and abs(diff.area_err_pct) <= self.config.area_tol_pct
            and diff.genus_diff == 0
        )

    def _format_diag_report(
        self,
        telemetry: CadTelemetry,
        diff: FactsDiff,
        view_eval: ViewEvalResult,
        facts: dict[str, Any] | None,
        exec_note: str = "",
    ) -> str:
        diag_lines = diff.summary_lines(telemetry, facts or {})
        diag_lines.append(
            f"- Visual 8-view Silhouette IoU: mean={view_eval.mean_silhouette_iou:.3f}, "
            f"min={view_eval.min_silhouette_iou:.3f}, edge_F1={view_eval.mean_edge_f1:.3f}"
        )
        if view_eval.worst_views:
            worst_str = ", ".join(
                f"{m.name} (IoU={m.silhouette_iou:.2f}, Edge={m.edge_f1:.2f})"
                for m in view_eval.worst_views[:3]
            )
            diag_lines.append(f"- Worst matching views: {worst_str}")
        if view_eval.handedness_flip_detected:
            diag_lines.append(
                "- HANDEDNESS / MIRROR ERROR DETECTED: Reflected silhouette matches better in multiple views; "
                "check coordinate sign / mirroring!"
            )
        if exec_note:
            diag_lines.append(f"- Note: {exec_note}")
        return "\n".join(diag_lines)

    def _execute_tool_call(
        self,
        call: dict[str, Any],
        views: list[Path],
        facts: dict[str, Any] | None,
        branch_dir: Path,
        probe_idx: int,
    ) -> tuple[str, bytes | None, ProbeOutcome | None]:
        name = call.get("name", "")
        args, err = _normalize_tool_args(call.get("arguments"))
        if err is not None:
            return err, None, None

        if name == "measure_ortho_features":
            view_name = str(args.get("view", "")).strip()
            detail = str(args.get("detail", "summary")).strip().lower()
            if detail not in {"summary", "full"}:
                detail = "summary"
            return measure_ortho_features(views, facts, view_name, detail=detail), None, None

        if name == "zoom_view_crop":
            view_name = str(args.get("view", "")).strip()
            try:
                legend, png_bytes = zoom_view_crop(
                    views,
                    facts,
                    view_name=view_name,
                    h_min=args.get("h_min"),
                    v_min=args.get("v_min"),
                    h_max=args.get("h_max"),
                    v_max=args.get("v_max"),
                )
                return legend, png_bytes, None
            except Exception as exc:
                return f"zoom_view_crop failed: {exc}", None, None

        if name == "cad_probe":
            probe_code = str(args.get("code", ""))
            probe = run_cad_probe(
                code=probe_code,
                views=views,
                facts=facts,
                work_dir=branch_dir,
                probe_idx=probe_idx,
                exec_timeout_s=self.config.exec_timeout_s,
                render_eval_px=min(256, self.config.render_eval_px),
            )
            return probe.report, None, probe

        return f"Unknown tool `{name}`; valid tools are measure_ortho_features, zoom_view_crop, cad_probe.", None, None

    def _run_turn_branch(
        self,
        *,
        case_name: str,
        turn: int,
        branch_label: str,
        prompt_messages: list[dict[str, Any]],
        branch_dir: Path,
        views: list[Path],
        facts: dict[str, Any] | None,
        turn_effort: str,
        temperature: float,
        seed: int,
        global_deadline: float | None,
        allow_tools: bool,
    ) -> _BranchOutcome:
        branch_dir.mkdir(parents=True, exist_ok=True)
        legacy_patched = self._is_call_llm_monkeypatched()
        with self._state_lock:
            tools_active = allow_tools and self._tools_supported and not legacy_patched and self.config.max_tool_steps > 0

        max_steps = min(self.config.max_tool_steps, 2 if turn > 0 else self.config.max_tool_steps) if tools_active else 0
        turn_msgs: list[dict[str, Any]] = list(prompt_messages)
        probe_outcomes: list[ProbeOutcome] = []
        probe_counter = 0
        raw_reply = ""
        stepdown_effort: str | None = None
        last_probe_code_fallback = ""

        total_iters = (max_steps + 1) if tools_active else 1
        for step in range(total_iters):
            rem = (global_deadline - time.monotonic()) if global_deadline is not None else None
            if rem is not None and rem < 12.0:
                return _BranchOutcome(ok=False, llm_failed=True, stepdown_effort=stepdown_effort)

            is_final_step = (not tools_active) or (step >= max_steps) or (rem is not None and rem < 40.0)
            # Keep schema on step > 0 so system-prompt KV prefix cache stays intact when role="tool" history exists
            step_tools = CAD_TOOL_SCHEMAS if (tools_active and (not is_final_step or step > 0)) else None
            step_msgs = list(turn_msgs)
            if tools_active and step > 0 and is_final_step:
                final_prompt = (
                    "Tool budget for this turn is complete. Output ONLY the final complete CadQuery 2.8 "
                    "Python script inside one ```python ... ``` code fence assigning the single solid to `result`."
                )
                if step_msgs and step_msgs[-1].get("role") == "user":
                    last_c = step_msgs[-1].get("content")
                    if isinstance(last_c, list):
                        step_msgs[-1] = {
                            "role": "user",
                            "content": [*last_c, {"type": "text", "text": final_prompt}],
                        }
                    else:
                        step_msgs[-1] = {"role": "user", "content": f"{last_c}\n\n{final_prompt}"}
                else:
                    step_msgs.append({"role": "user", "content": final_prompt})

            call_timeout = self.config.llm_timeout_s
            if rem is not None:
                call_timeout = min(call_timeout, max(15.0, rem - 10.0))

            try:
                if legacy_patched:
                    raw_reply = self._call_llm(step_msgs, reasoning_effort=turn_effort, timeout_s=call_timeout)
                    parsed_calls: list[dict[str, Any]] = []
                    raw_content = raw_reply
                    raw_assistant_tcs = None
                else:
                    raw_reply, parsed_calls, raw_content, raw_assistant_tcs = self._invoke_completion(
                        messages=step_msgs,
                        reasoning_effort=turn_effort,
                        timeout_s=call_timeout,
                        tools=step_tools,
                        temperature=temperature,
                        seed=seed,
                    )
            except (APITimeoutError, TimeoutError) as exc:
                if turn_effort == "medium":
                    stepdown_effort = "low"
                elif turn_effort == "low":
                    stepdown_effort = "none"
                rem_after = (global_deadline - time.monotonic()) if global_deadline is not None else 160.0
                if turn_effort != "none" and self.config.thinking_mode != "off" and rem_after >= 30.0:
                    logger.warning(
                        "[%s] turn %d%s timed out (%s); retrying instant non-thinking pass (next effort=%s).",
                        case_name,
                        turn,
                        branch_label,
                        exc,
                        stepdown_effort or turn_effort,
                    )
                    try:
                        if legacy_patched:
                            raw_reply = self._call_llm(
                                step_msgs,
                                reasoning_effort="none",
                                timeout_s=min(150.0, rem_after - 10.0),
                            )
                        else:
                            raw_reply, _, _, _ = self._invoke_completion(
                                messages=step_msgs,
                                reasoning_effort="none",
                                timeout_s=min(150.0, rem_after - 10.0),
                                tools=None,
                                temperature=temperature,
                                seed=seed,
                            )
                        parsed_calls = []
                        break
                    except Exception as exc2:
                        logger.warning("[%s] LLM non-thinking retry failed on turn %d%s: %s", case_name, turn, branch_label, exc2)
                        return _BranchOutcome(ok=False, llm_failed=True, stepdown_effort=stepdown_effort)
                logger.warning("[%s] LLM call timed out on turn %d%s: %s", case_name, turn, branch_label, exc)
                return _BranchOutcome(ok=False, llm_failed=True, stepdown_effort=stepdown_effort)
            except Exception as exc:
                logger.warning("[%s] LLM call failed on turn %d%s: %s", case_name, turn, branch_label, exc)
                return _BranchOutcome(ok=False, llm_failed=True, stepdown_effort=stepdown_effort)

            if not parsed_calls:
                break

            if is_final_step:
                # Harvest any cad_probe code argument if final message gave no fenced python
                for tc in parsed_calls:
                    if tc.get("name") == "cad_probe":
                        p_args, _ = _normalize_tool_args(tc.get("arguments"))
                        if isinstance(p_args.get("code"), str) and p_args["code"].strip():
                            last_probe_code_fallback = p_args["code"].strip()
                break

            # Execute bounded tool calls (max 4 per step; still reply to every id for OpenAI compliance)
            max_per_step = 4
            tool_results: list[tuple[str, str, bytes | None]] = []
            early_probe: ProbeOutcome | None = None
            for idx, tc in enumerate(parsed_calls):
                if idx >= max_per_step:
                    tool_results.append((tc["id"], "Skipped (max 4 tool calls per step); proceed with code.", None))
                    continue
                if tc.get("name") == "cad_probe":
                    probe_counter += 1
                    p_args, _ = _normalize_tool_args(tc.get("arguments"))
                    if isinstance(p_args.get("code"), str) and p_args["code"].strip():
                        last_probe_code_fallback = p_args["code"].strip()
                txt_out, png_bytes, probe = self._execute_tool_call(tc, views, facts, branch_dir, probe_counter)
                tool_results.append((tc["id"], txt_out, png_bytes))
                if probe is not None and probe.ok and probe.score is not None:
                    probe_outcomes.append(probe)
                    if self._should_early_exit(probe.diff, probe.view_eval, probe.score, facts):
                        early_probe = probe

            if early_probe is not None and early_probe.step_path is not None:
                tel, diff, ve, sc = (
                    early_probe.telemetry,
                    early_probe.diff,
                    early_probe.view_eval,
                    early_probe.score,
                )
                if self.config.render_eval_px > 256:
                    try:
                        tel, diff, ve, sc = self._evaluate_step(early_probe.step_path, facts, views)
                    except Exception:
                        pass
                diag_rep = self._format_diag_report(tel, diff, ve, facts)
                return _BranchOutcome(
                    ok=True,
                    code=early_probe.code,
                    step_path=early_probe.step_path,
                    telemetry=tel,
                    diff=diff,
                    view_eval=ve,
                    score=sc,
                    diag_report=diag_rep,
                    stepdown_effort=stepdown_effort,
                    early_exit=True,
                )

            # Append assistant + tool messages in appropriate native/text dialect
            if raw_assistant_tcs is not None:
                turn_msgs.append(
                    {
                        "role": "assistant",
                        "content": raw_reply or "",
                        "tool_calls": raw_assistant_tcs,
                    }
                )
                image_parts: list[dict[str, Any]] = []
                for cid, txt, png_bytes in tool_results:
                    turn_msgs.append({"role": "tool", "tool_call_id": cid, "content": txt})
                    if png_bytes is not None:
                        image_parts.append(
                            {"type": "image_url", "image_url": {"url": _png_bytes_to_data_url(png_bytes)}}
                        )
                if image_parts:
                    turn_msgs.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Annotated native-resolution zoomed view crop(s):"},
                                *image_parts,
                            ],
                        }
                    )
            else:
                turn_msgs.append({"role": "assistant", "content": raw_content})
                combined_txt = "\n\n".join(f"[Tool Output #{i + 1}]\n{txt}" for i, (_, txt, _) in enumerate(tool_results))
                user_parts: list[dict[str, Any]] = [{"type": "text", "text": combined_txt}]
                for _, _, png_bytes in tool_results:
                    if png_bytes is not None:
                        user_parts.append(
                            {"type": "image_url", "image_url": {"url": _png_bytes_to_data_url(png_bytes)}}
                        )
                turn_msgs.append({"role": "user", "content": user_parts})

        # Evaluate final response code (and compare against any verified cad_probe candidates)
        code = extract_python_code(raw_reply)
        if not code and last_probe_code_fallback:
            code = extract_python_code(last_probe_code_fallback)

        best_probe: ProbeOutcome | None = None
        if probe_outcomes:
            best_probe = max(probe_outcomes, key=lambda p: p.score.total if p.score else -1.0)

        final_outcome: _BranchOutcome | None = None
        if code:
            matching_probe = next((p for p in reversed(probe_outcomes) if p.code.strip() == code.strip()), None)
            if matching_probe is not None and matching_probe.score is not None:
                # Re-evaluate at full render_eval_px if probe used smaller eval_px
                if matching_probe.step_path is not None and self.config.render_eval_px > 256:
                    try:
                        tel, diff, ve, sc = self._evaluate_step(matching_probe.step_path, facts, views)
                    except Exception:
                        tel, diff, ve, sc = (
                            matching_probe.telemetry,
                            matching_probe.diff,
                            matching_probe.view_eval,
                            matching_probe.score,
                        )
                else:
                    tel, diff, ve, sc = (
                        matching_probe.telemetry,
                        matching_probe.diff,
                        matching_probe.view_eval,
                        matching_probe.score,
                    )
                diag_rep = self._format_diag_report(tel, diff, ve, facts)
                final_outcome = _BranchOutcome(
                    ok=True,
                    code=code,
                    step_path=matching_probe.step_path,
                    telemetry=tel,
                    diff=diff,
                    view_eval=ve,
                    score=sc,
                    diag_report=diag_rep,
                    stepdown_effort=stepdown_effort,
                    early_exit=self._should_early_exit(diff, ve, sc, facts),
                )
            else:
                cand_step = branch_dir / "final.stp"
                exec_res = execute_cadquery_script(
                    code,
                    cand_step,
                    timeout_s=self.config.exec_timeout_s,
                    allow_fillet_strip_fallback=True,
                )
                if exec_res.ok and exec_res.step_path is not None:
                    try:
                        tel, diff, ve, sc = self._evaluate_step(exec_res.step_path, facts, views)
                        exec_note = exec_res.error if exec_res.used_fillet_fallback else ""
                        diag_rep = self._format_diag_report(tel, diff, ve, facts, exec_note=exec_note)
                        final_outcome = _BranchOutcome(
                            ok=True,
                            code=code,
                            step_path=exec_res.step_path,
                            telemetry=tel,
                            diff=diff,
                            view_eval=ve,
                            score=sc,
                            diag_report=diag_rep,
                            used_fillet_fallback=exec_res.used_fillet_fallback,
                            exec_note=exec_note,
                            stepdown_effort=stepdown_effort,
                            early_exit=self._should_early_exit(diff, ve, sc, facts),
                        )
                    except Exception as exc:
                        final_outcome = _BranchOutcome(
                            ok=False,
                            code=code,
                            diag_report=f"Generated STEP failed post-export inspection: {exc}",
                            stepdown_effort=stepdown_effort,
                        )
                else:
                    tb_tail = "\n".join(exec_res.traceback_text.splitlines()[-15:])
                    err_report = f"CadQuery script failed to execute:\n{exec_res.error}"
                    if tb_tail:
                        err_report += f"\nTraceback (tail):\n{tb_tail}"
                    logger.info("[%s] turn %d%s exec failed: %s", case_name, turn, branch_label, exec_res.error.splitlines()[0])
                    final_outcome = _BranchOutcome(
                        ok=False,
                        code=code,
                        diag_report=err_report,
                        stepdown_effort=stepdown_effort,
                    )

        # Promote higher-scoring verified probe if final code failed or regressed
        if best_probe is not None and best_probe.score is not None and best_probe.step_path is not None:
            if final_outcome is None or not final_outcome.ok or (
                final_outcome.score is not None and best_probe.score.total > final_outcome.score.total
            ):
                p_tel, p_diff, p_ve, p_sc = (
                    best_probe.telemetry,
                    best_probe.diff,
                    best_probe.view_eval,
                    best_probe.score,
                )
                if self.config.render_eval_px > 256:
                    try:
                        p_tel, p_diff, p_ve, p_sc = self._evaluate_step(best_probe.step_path, facts, views)
                    except Exception:
                        pass
                diag_rep = self._format_diag_report(p_tel, p_diff, p_ve, facts)
                return _BranchOutcome(
                    ok=True,
                    code=best_probe.code,
                    step_path=best_probe.step_path,
                    telemetry=p_tel,
                    diff=p_diff,
                    view_eval=p_ve,
                    score=p_sc,
                    diag_report=diag_rep,
                    stepdown_effort=stepdown_effort,
                    early_exit=self._should_early_exit(p_diff, p_ve, p_sc, facts),
                )

        if final_outcome is not None:
            return final_outcome
        return _BranchOutcome(
            ok=False,
            code=code,
            diag_report="CadQuery script failed to execute:\nGenerated script is empty.",
            stepdown_effort=stepdown_effort,
        )

    def generate_cad(
        self,
        prompt: str,
        views: list[Path],
        context: Path | None,
        output_path: Path,
        *,
        facts: dict[str, Any] | None = None,
        case_name: str | None = None,
        global_deadline: float | None = None,
    ) -> GenerationSummary:
        t_start = time.monotonic()
        name = case_name or output_path.stem
        effort, max_turns = self._plan_budget(facts)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with self._state_lock:
            tools_for_prompt = self._tools_supported and not self._is_call_llm_monkeypatched()

        initial_user_content = build_initial_user_content(
            case_name=name,
            description=prompt,
            facts=facts,
            views=views,
            context=context,
            tile_px=self.config.contact_tile_px,
            context_px=self.config.context_max_px,
            enable_tools=tools_for_prompt,
        )

        best_score: SurrogateScore | None = None
        best_diff: Any = None
        best_ve: Any = None
        best_step_tmp: Path | None = None
        best_code: str | None = None
        best_diag: str | None = None
        best_diff_png: bytes | None = None

        last_code = ""
        last_diag = ""
        last_diff_png: bytes | None = None
        turns_used = 0
        used_fallback = False
        prev_exec_failed = False
        stagnant_streak = 0
        no_improve_streak = 0

        with tempfile.TemporaryDirectory(prefix=f"cad_work_{name}_") as work_str:
            work_dir = Path(work_str)

            # Build & write deterministic CPU visual-hull floor upfront so a crash or deadline never leaves an invalid/missing file
            floor_step: Path | None = work_dir / "fallback_hull.stp"
            floor_score: SurrogateScore | None = None
            floor_diff: Any = None
            try:
                if generate_fallback_step(views, facts, floor_step):
                    _atomic_copy_file(floor_step, output_path)
                    _, floor_diff, floor_ve, floor_score = self._evaluate_step(floor_step, facts, views)
                    logger.info(
                        "[%s] Floor hull ready (score=%.3f, iou=%.3f)",
                        name,
                        floor_score.total,
                        floor_ve.mean_silhouette_iou,
                    )
                else:
                    floor_step = None
            except Exception as exc:
                logger.warning("[%s] Floor evaluation warning: %s", name, exc)
                if not floor_step.is_file():
                    floor_step = None

            for turn in range(max_turns):
                remaining = (global_deadline - time.monotonic()) if global_deadline is not None else None
                if remaining is not None and remaining < 12.0:
                    logger.warning("[%s] Global budget nearly exhausted (%.1fs left); finalizing.", name, remaining)
                    break

                turns_used = turn + 1
                with self._state_lock:
                    turn_tools_enabled = self._tools_supported and not self._is_call_llm_monkeypatched()
                if remaining is not None and remaining < 60.0:
                    turn_tools_enabled = False

                if turn == 0:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": initial_user_content},
                    ]
                else:
                    messages = build_repair_messages(
                        initial_user_content=initial_user_content,
                        last_code=last_code,
                        diagnostic_report=last_diag,
                        diff_composite_png=last_diff_png,
                        enable_tools=turn_tools_enabled,
                    )

                if effort == "none":
                    turn_effort = "none"
                elif turn == 0:
                    turn_effort = effort
                elif turn >= 2 and not prev_exec_failed:
                    turn_effort = "none" if effort == "low" else "low"
                else:
                    turn_effort = "medium" if effort == "xhigh" else effort

                if remaining is not None:
                    if remaining < 60.0:
                        turn_effort = "none"
                    elif remaining < 180.0 and turn_effort != "none":
                        turn_effort = "low"

                # Parallel Best-of-K on Turn 0 when configured & sufficient global budget remains
                k_samples = 1
                if (
                    turn == 0
                    and self.config.best_of_k > 1
                    and (remaining is None or remaining >= min(180.0, 0.35 * self.config.global_timeout_s))
                ):
                    k_samples = self.config.best_of_k

                outcomes: list[_BranchOutcome] = []
                if k_samples == 1:
                    outcomes.append(
                        self._run_turn_branch(
                            case_name=name,
                            turn=turn,
                            branch_label="",
                            prompt_messages=messages,
                            branch_dir=work_dir / f"turn_{turn}_k0",
                            views=views,
                            facts=facts,
                            turn_effort=turn_effort,
                            temperature=self.config.temperature,
                            seed=self.config.seed,
                            global_deadline=global_deadline,
                            allow_tools=turn_tools_enabled,
                        )
                    )
                else:
                    alt_temp = (
                        max(0.25, round(self.config.temperature - 0.3, 2))
                        if self.config.temperature >= 0.7
                        else min(1.0, round(self.config.temperature + 0.3, 2))
                    )
                    temp_schedule = [self.config.temperature, alt_temp, 0.35, 0.95]
                    with ThreadPoolExecutor(max_workers=k_samples) as branch_pool:
                        futs = [
                            branch_pool.submit(
                                self._run_turn_branch,
                                case_name=name,
                                turn=turn,
                                branch_label=f"#k{k_idx}",
                                prompt_messages=messages,
                                branch_dir=work_dir / f"turn_{turn}_k{k_idx}",
                                views=views,
                                facts=facts,
                                turn_effort=turn_effort,
                                temperature=temp_schedule[k_idx % len(temp_schedule)],
                                seed=self.config.seed + k_idx,
                                global_deadline=global_deadline,
                                allow_tools=turn_tools_enabled,
                            )
                            for k_idx in range(k_samples)
                        ]
                        for fut in futs:
                            outcomes.append(fut.result())

                for out in outcomes:
                    if out.stepdown_effort == "none" or (out.stepdown_effort == "low" and effort == "medium"):
                        effort = out.stepdown_effort

                valid_outcomes = [o for o in outcomes if o.ok and o.score is not None and o.step_path is not None]
                if not valid_outcomes:
                    non_llm_errs = [o for o in outcomes if not o.llm_failed]
                    if not non_llm_errs:
                        break
                    chosen_err = non_llm_errs[0]
                    prev_exec_failed = True
                    last_code = chosen_err.code or best_code or ""
                    last_diag = chosen_err.diag_report
                    last_diff_png = best_diff_png
                    continue

                winner = max(valid_outcomes, key=lambda o: o.score.total if o.score else -1.0)
                telemetry = winner.telemetry
                diff = winner.diff
                view_eval = winner.view_eval
                score = winner.score
                cand_step = winner.step_path
                code = winner.code
                diag_report = winner.diag_report

                assert telemetry is not None and diff is not None and view_eval is not None and score is not None
                prev_exec_failed = False

                diff_png = build_visual_diff_composite(
                    view_eval,
                    worst_n=2,
                    tile_px=self.config.diff_tile_px,
                )

                logger.info(
                    "[%s] turn %d%s valid=%s solids=%d score=%.3f iou=%.3f vol_err=%+.1f%% genus=%d",
                    name,
                    turn,
                    f" (best-of-{k_samples})" if k_samples > 1 else "",
                    telemetry.is_valid,
                    telemetry.solids,
                    score.total,
                    view_eval.mean_silhouette_iou,
                    diff.vol_err_pct,
                    telemetry.through_holes_genus,
                )

                prev_best_score = best_score
                prev_best_diff = best_diff
                prev_best_code = best_code
                is_same = False
                if prev_best_score is None or score.total > prev_best_score.total + 0.002:
                    stagnant_streak = 0
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
                    is_same = (code.strip() == (prev_best_code or "").strip()) or (
                        prev_best_diff is not None
                        and abs(score.total - prev_best_score.total) <= 0.002
                        and abs(diff.vol_err_pct - prev_best_diff.vol_err_pct) <= 0.25
                    )
                    if is_same:
                        stagnant_streak += 1
                    else:
                        stagnant_streak = 0

                stagnation_note = ""
                if is_same and prev_best_score is not None and prev_best_diff is not None:
                    stagnation_note = (
                        f"STAGNATION WARNING: Your last edit produced virtually identical score/geometry "
                        f"to the previous best (score={prev_best_score.total:.3f}, vol_err={prev_best_diff.vol_err_pct:+.1f}%). "
                        f"Do NOT repeat the same script — materially change the structural geometry, "
                        f"hole/boss coordinates, or profile cross-sections to fix the discrepancies below.\n"
                    )

                if best_score is None or score.total > best_score.total:
                    best_score = score
                    best_diff = diff
                    best_ve = view_eval
                    best_step_tmp = cand_step
                    best_code = code
                    best_diag = diag_report
                    best_diff_png = diff_png
                    last_code = code
                    last_diag = f"{stagnation_note}{diag_report}" if stagnation_note else diag_report
                    last_diff_png = diff_png
                    if _prefer_llm_over_floor(best_score, best_diff, best_ve, floor_score, floor_diff):
                        _atomic_copy_file(best_step_tmp, output_path)
                else:
                    # Revert base code in prompt back to best candidate so repair doesn't spiral
                    last_code = best_code or code
                    last_diag = (
                        f"{stagnation_note}"
                        f"Your previous edit did not improve score ({score.total:.3f} <= best {best_score.total:.3f}). "
                        f"Reverting to the previous best script.\n"
                        f"Latest attempt feedback:\n{diag_report}\n"
                        f"Previous best candidate telemetry:\n{best_diag or ''}"
                    )
                    last_diff_png = best_diff_png

                if winner.early_exit or self._should_early_exit(diff, view_eval, score, facts):
                    logger.info("[%s] Early-exit convergence achieved at turn %d (score=%.3f).", name, turn, score.total)
                    break

                if turn >= 2 and (stagnant_streak >= 2 or no_improve_streak >= 3):
                    logger.info(
                        "[%s] Stopping repair early at turn %d due to plateau (best score=%.3f, stagnant=%d).",
                        name,
                        turn,
                        best_score.total if best_score else 0.0,
                        stagnant_streak,
                    )
                    break

            # Select best verified candidate between LLM solid and deterministic visual-hull floor
            choose_llm = _prefer_llm_over_floor(best_score, best_diff, best_ve, floor_score, floor_diff)

            if choose_llm and best_step_tmp is not None and best_step_tmp.is_file():
                _atomic_copy_file(best_step_tmp, output_path)
                final_score = best_score
            elif floor_step is not None and floor_step.is_file():
                _atomic_copy_file(floor_step, output_path)
                final_score = floor_score
                used_fallback = True
            else:
                used_fallback = generate_fallback_step(views, facts, output_path)
                final_score = best_score

        elapsed = time.monotonic() - t_start
        return GenerationSummary(
            case_name=name,
            output_path=output_path,
            turns_used=turns_used,
            score=final_score.total if final_score else 0.0,
            mean_iou=final_score.mean_silhouette_iou if final_score else 0.0,
            used_fallback=used_fallback,
            elapsed_s=round(elapsed, 2),
        )


_ORIGINAL_CALL_LLM = QwenCadAgent._call_llm
