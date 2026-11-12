from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI

from agentic_cad.cadquery_ref import extract_python_code
from agentic_cad.config import HarnessConfig
from agentic_cad.executor import execute_cadquery_script
from agentic_cad.fallback import generate_fallback_step
from agentic_cad.inspector import compare_against_facts, inspect_shape, load_step_shape
from agentic_cad.prompts import (
    SYSTEM_PROMPT,
    build_initial_user_content,
    build_repair_messages,
)
from agentic_cad.renderer import build_visual_diff_composite, evaluate_views, tessellate_shape
from agentic_cad.scorer import SurrogateScore, score_candidate

logger = logging.getLogger(__name__)


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
            max_retries=1,
        )

    def _plan_budget(self, facts: dict[str, Any] | None) -> tuple[str, int]:
        """Choose reasoning effort and max repair turns from BRep face complexity (no per-part timer)."""
        faces = int((facts or {}).get("faces") or 50)
        if faces <= 30:
            return "medium", self.config.max_turns_simple
        if faces <= 200:
            return "high", self.config.max_turns_medium
        return "high", self.config.max_turns_complex

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        reasoning_effort: str,
        timeout_s: float,
    ) -> str:
        t_mode = self.config.thinking_mode
        kwargs: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
            "timeout": timeout_s,
        }
        if t_mode == "off":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            kwargs["reasoning_effort"] = reasoning_effort

        def _invoke(call_kwargs: dict[str, Any]):
            try:
                return self.client.chat.completions.create(**call_kwargs)
            except BadRequestError as exc:
                err_s = str(exc)
                if "reasoning_effort" in err_s and "reasoning_effort" in call_kwargs:
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
        if t_mode == "auto" and (not content or (finish_reason == "length" and "```" not in content)):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("reasoning_effort", None)
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
        if not facts:
            return view_eval.mean_silhouette_iou >= self.config.iou_early_exit
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

        with tempfile.TemporaryDirectory(prefix=f"cad_work_{name}_") as work_str:
            work_dir = Path(work_str)

            # Build & write deterministic CPU visual-hull floor upfront so a crash or deadline never leaves an invalid/missing file
            floor_step: Path | None = work_dir / "fallback_hull.stp"
            floor_score: SurrogateScore | None = None
            floor_diff: Any = None
            try:
                if generate_fallback_step(views, facts, floor_step):
                    shutil.copy2(floor_step, output_path)
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
                if global_deadline is not None:
                    remaining = global_deadline - time.monotonic()
                    if remaining < 12.0:
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

                call_timeout = self.config.llm_timeout_s
                if global_deadline is not None:
                    call_timeout = min(call_timeout, max(15.0, global_deadline - time.monotonic() - 10.0))

                try:
                    raw_reply = self._call_llm(messages, reasoning_effort=effort, timeout_s=call_timeout)
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
                    last_code = code
                    last_diag = f"Generated STEP failed post-export inspection: {exc}"
                    last_diff_png = best_diff_png
                    continue

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

                if best_score is None or score.total > best_score.total:
                    best_score = score
                    best_diff = diff
                    best_ve = view_eval
                    best_step_tmp = cand_step
                    best_code = code
                    best_diag = diag_report
                    best_diff_png = diff_png
                    last_code = code
                    last_diag = diag_report
                    last_diff_png = diff_png
                else:
                    # Revert base code in prompt back to best candidate so repair doesn't spiral
                    last_code = best_code or code
                    last_diag = (
                        f"Your previous edit lowered score ({score.total:.3f} < best {best_score.total:.3f}). "
                        f"Reverting to the previous best script.\n"
                        f"Latest attempt feedback:\n{diag_report}\n"
                        f"Previous best candidate telemetry:\n{best_diag or ''}"
                    )
                    last_diff_png = best_diff_png

                if self._should_early_exit(diff, view_eval, score, facts):
                    logger.info("[%s] Early-exit convergence achieved at turn %d (score=%.3f).", name, turn, score.total)
                    break

            # Select best verified candidate between LLM solid and deterministic visual-hull floor
            choose_llm = False
            if best_step_tmp is not None and best_score is not None and best_score.is_valid_single_solid:
                if floor_score is None or best_score.total >= floor_score.total:
                    choose_llm = True
                elif (
                    best_diff is not None
                    and floor_diff is not None
                    and best_ve is not None
                    and best_diff.genus_diff == 0
                    and best_diff.max_bbox_err_pct <= 2.0
                    and best_ve.mean_silhouette_iou >= 0.80
                    and abs(best_diff.vol_err_pct) < 0.4 * abs(floor_diff.vol_err_pct)
                    and best_score.total >= floor_score.total - 0.05
                ):
                    choose_llm = True

            if choose_llm and best_step_tmp is not None and best_step_tmp.is_file():
                shutil.copy2(best_step_tmp, output_path)
                final_score = best_score
            elif floor_step is not None and floor_step.is_file():
                shutil.copy2(floor_step, output_path)
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
