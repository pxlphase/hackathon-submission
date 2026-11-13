from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APITimeoutError, BadRequestError, OpenAI

from agentic_cad.cadquery_ref import extract_python_code
from agentic_cad.config import HarnessConfig
from agentic_cad.executor import execute_cadquery_script
from agentic_cad.fallback import generate_fallback_step
from agentic_cad.inspector import FactsDiff, compare_against_facts, inspect_shape, load_step_shape
from agentic_cad.prompts import (
    SYSTEM_PROMPT,
    build_initial_user_content,
    build_repair_messages,
)
from agentic_cad.renderer import ViewEvalResult, build_visual_diff_composite, evaluate_views, tessellate_shape
from agentic_cad.scorer import SurrogateScore, score_candidate

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

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        reasoning_effort: str,
        timeout_s: float,
    ) -> str:
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
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
            "timeout": timeout_s,
        }
        if t_mode == "off" or reasoning_effort == "none":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            kwargs["reasoning_effort"] = reasoning_effort

        def _invoke(call_kwargs: dict[str, Any]):
            try:
                return self.client.chat.completions.create(**call_kwargs)
            except BadRequestError as exc:
                err_s = str(exc).lower()
                if "reasoning" in err_s and "reasoning_effort" in call_kwargs:
                    if call_kwargs["reasoning_effort"] != "medium" and "medium" in err_s:
                        call_kwargs["reasoning_effort"] = "medium"
                        try:
                            return self.client.chat.completions.create(**call_kwargs)
                        except BadRequestError:
                            pass
                    call_kwargs.pop("reasoning_effort", None)
                    return self.client.chat.completions.create(**call_kwargs)
                if "chat_template_kwargs" in err_s and "extra_body" in call_kwargs:
                    call_kwargs.pop("extra_body", None)
                    return self.client.chat.completions.create(**call_kwargs)
                raise

        resp = _invoke(dict(kwargs))
        choice = resp.choices[0]
        msg = choice.message
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
            if "```" in reasoning:
                content = reasoning

        # Guard against thinking-budget starvation returning empty/cutoff reply: retry once with thinking disabled
        finish_reason = getattr(choice, "finish_reason", None)
        if (
            t_mode == "auto"
            and reasoning_effort != "none"
            and (not content or (finish_reason == "length" and "```" not in content))
        ):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("reasoning_effort", None)
            retry_kwargs["max_tokens"] = min(int(retry_kwargs["max_tokens"]), 8192)
            retry_kwargs["timeout"] = min(150.0, timeout_s)
            retry_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            try:
                resp2 = _invoke(retry_kwargs)
                content2 = (getattr(resp2.choices[0].message, "content", None) or "").strip()
                if content2:
                    content = content2
            except Exception:
                pass
        return content

    def _evaluate_step(
        self,
        step_path: Path,
        facts: dict[str, Any] | None,
        views: list[Path],
    ) -> tuple[Any, Any, Any, SurrogateScore]:
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

        initial_user_content = build_initial_user_content(
            case_name=name,
            description=prompt,
            facts=facts,
            views=views,
            context=context,
            tile_px=self.config.contact_tile_px,
            context_px=self.config.context_max_px,
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

                call_timeout = self.config.llm_timeout_s
                if remaining is not None:
                    call_timeout = min(call_timeout, max(15.0, remaining - 10.0))

                try:
                    raw_reply = self._call_llm(messages, reasoning_effort=turn_effort, timeout_s=call_timeout)
                except (APITimeoutError, TimeoutError) as exc:
                    if turn_effort == "medium":
                        effort = "low"
                    elif turn_effort == "low":
                        effort = "none"
                    rem_after = (global_deadline - time.monotonic()) if global_deadline is not None else 160.0
                    if turn_effort != "none" and self.config.thinking_mode != "off" and rem_after >= 30.0:
                        logger.warning(
                            "[%s] turn %d timed out (%s); retrying instant non-thinking pass (next effort=%s).",
                            name,
                            turn,
                            exc,
                            effort,
                        )
                        try:
                            raw_reply = self._call_llm(
                                messages,
                                reasoning_effort="none",
                                timeout_s=min(150.0, rem_after - 10.0),
                            )
                        except Exception as exc2:
                            logger.warning("[%s] LLM non-thinking retry failed on turn %d: %s", name, turn, exc2)
                            break
                    else:
                        logger.warning("[%s] LLM call timed out on turn %d: %s", name, turn, exc)
                        break
                except Exception as exc:
                    logger.warning("[%s] LLM call failed on turn %d: %s", name, turn, exc)
                    break

                code = extract_python_code(raw_reply)
                cand_step = work_dir / f"cand_t{turn}.stp"
                exec_res = execute_cadquery_script(
                    code,
                    cand_step,
                    timeout_s=self.config.exec_timeout_s,
                    allow_fillet_strip_fallback=True,
                )

                if not exec_res.ok or exec_res.step_path is None:
                    prev_exec_failed = True
                    tb_tail = "\n".join(exec_res.traceback_text.splitlines()[-15:])
                    err_report = f"CadQuery script failed to execute:\n{exec_res.error}"
                    if tb_tail:
                        err_report += f"\nTraceback (tail):\n{tb_tail}"
                    last_code = code or best_code or ""
                    last_diag = err_report
                    last_diff_png = best_diff_png
                    logger.info("[%s] turn %d exec failed: %s", name, turn, exec_res.error.splitlines()[0])
                    continue

                try:
                    telemetry, diff, view_eval, score = self._evaluate_step(exec_res.step_path, facts, views)
                except Exception as exc:
                    prev_exec_failed = True
                    last_code = code
                    last_diag = f"Generated STEP failed post-export inspection: {exc}"
                    last_diff_png = best_diff_png
                    continue

                prev_exec_failed = False
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
                if exec_res.used_fillet_fallback:
                    diag_lines.append(f"- Note: {exec_res.error}")
                diag_report = "\n".join(diag_lines)

                diff_png = build_visual_diff_composite(
                    view_eval,
                    worst_n=2,
                    tile_px=self.config.diff_tile_px,
                )

                logger.info(
                    "[%s] turn %d valid=%s solids=%d score=%.3f iou=%.3f vol_err=%+.1f%% genus=%d",
                    name,
                    turn,
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

                if self._should_early_exit(diff, view_eval, score, facts):
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
