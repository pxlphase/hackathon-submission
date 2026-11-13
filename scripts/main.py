import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_cad.agent import QwenCadAgent
from agentic_cad.config import HarnessConfig
from agentic_cad.fallback import generate_fallback_step

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = REPO_ROOT / "artifacts/inputs"
DEFAULT_OUTPUTS = REPO_ROOT / "artifacts/outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    name: str  # test_case_<n>
    description: str  # contents of description.md
    views: list[Path]  # images/view_0_front.png ... view_7_iso_back_left_bottom.png
    context: Path | None  # context.png, only present for parts taken from an assembly
    facts: dict[str, Any] | None = None  # parsed contents of facts.json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUTS)

    return parser.parse_args()


def load_test_cases(input_dir: Path) -> list[TestCase]:
    test_cases = []
    for case_dir in input_dir.glob("test_case_*"):
        if not (case_dir / "description.md").is_file():
            continue
        context = case_dir / "context.png"
        facts_path = case_dir / "facts.json"
        facts: dict[str, Any] | None = None
        if facts_path.is_file():
            try:
                facts = json.loads(facts_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", facts_path, exc)
        test_cases.append(TestCase(
            name=case_dir.name,
            description=(case_dir / "description.md").read_text(encoding="utf-8"),
            views=sorted((case_dir / "images").glob("view_*.png")),
            context=context if context.is_file() else None,
            facts=facts,
        ))
    return sorted(test_cases, key=lambda tc: int(tc.name.removeprefix("test_case_")))


def _process_one(
    agent: QwenCadAgent,
    tc: TestCase,
    output_dir: Path,
    global_deadline: float,
):
    output_path = output_dir / f"{tc.name}.stp"
    try:
        return agent.generate_cad(
            prompt=tc.description,
            views=tc.views,
            context=tc.context,
            output_path=output_path,
            facts=tc.facts,
            case_name=tc.name,
            global_deadline=global_deadline,
        )
    except Exception as exc:
        logger.exception("[%s] Unhandled error; invoking deterministic fallback: %s", tc.name, exc)
        generate_fallback_step(tc.views, tc.facts, output_path)
        return None


def main(args: argparse.Namespace) -> None:
    t0 = time.monotonic()
    cfg = HarnessConfig.from_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_cases(args.input_dir)
    if not test_cases:
        logger.warning("No test cases found in %s", args.input_dir)
        return

    global_deadline = t0 + cfg.global_timeout_s
    agent = QwenCadAgent(config=cfg)
    vlm_online = agent.resolve_endpoint_model()
    # Schedule simplest parts first so fast high-scoring parts finish upfront instead of starving at deadline
    ordered_cases = sorted(
        test_cases,
        key=lambda tc: int((tc.facts or {}).get("faces") or 50),
    )
    workers = min(cfg.max_workers, len(ordered_cases)) if vlm_online else 1
    logger.info(
        "Running %d test cases with workers=%d, model=%s, vlm_online=%s, global_budget=%.0fs",
        len(ordered_cases),
        workers,
        agent.config.openai_model,
        vlm_online,
        cfg.global_timeout_s,
    )

    summaries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, agent, tc, args.output_dir, global_deadline): tc
            for tc in ordered_cases
        }
        for fut in as_completed(futures):
            tc = futures[fut]
            res = fut.result()
            if res is not None:
                summaries.append(res)
                logger.info(
                    "Finished %s in %.1fs | turns=%d score=%.3f iou=%.3f fallback=%s",
                    res.case_name,
                    res.elapsed_s,
                    res.turns_used,
                    res.score,
                    res.mean_iou,
                    res.used_fallback,
                )

    total_elapsed = time.monotonic() - t0
    if summaries:
        avg_score = sum(s.score for s in summaries) / len(summaries)
        logger.info("All %d cases finished in %.1fs | mean surrogate score=%.3f", len(summaries), total_elapsed, avg_score)
    else:
        logger.info("Pipeline completed %d cases in %.1fs", len(ordered_cases), total_elapsed)


if __name__ == "__main__":
    args = parse_args()
    main(args)
