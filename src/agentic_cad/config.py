from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_if_present(env_path: Path | None = None) -> None:
    """Populate os.environ from a local .env file without overriding existing vars."""
    candidates = [env_path] if env_path is not None else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        break


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for the Qwen3.8-27B CAD generation & verification pipeline."""

    openai_base_url: str
    openai_api_key: str
    openai_model: str
    max_workers: int = 3
    global_timeout_s: float = 3300.0
    exec_timeout_s: float = 45.0
    llm_timeout_s: float = 300.0
    max_tokens: int = 16384
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 42
    thinking_mode: str = "auto"  # "auto" | "on" | "off"
    reasoning_effort: str = "auto"  # "auto" | "low" | "medium" | "none"

    # Convergence & Tier-1/Tier-2 thresholds
    iou_early_exit: float = 0.95
    vol_tol_pct: float = 2.0
    area_tol_pct: float = 3.0
    bbox_tol_pct: float = 1.5
    fallback_min_score: float = 0.25

    # Complexity-guided initial iteration budgets (never a hard per-part wall-clock cap)
    max_turns_simple: int = 4
    max_turns_medium: int = 5
    max_turns_complex: int = 6

    # Image packing sizes (aligned with Qwen3.8-27B 32x32 px vision tokens)
    contact_tile_px: int = 512
    context_max_px: int = 512
    diff_tile_px: int = 384
    render_eval_px: int = 384

    @classmethod
    def from_env(cls, load_dotenv: bool = True) -> HarnessConfig:
        if load_dotenv:
            load_dotenv_if_present()
        return cls(
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").strip()
            or "http://localhost:8000/v1",
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip() or "EMPTY",
            openai_model=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.8-27B").strip()
            or "Qwen/Qwen3.8-27B",
            max_workers=max(1, _env_int("CAD_MAX_WORKERS", 3)),
            global_timeout_s=max(60.0, _env_float("CAD_GLOBAL_TIMEOUT_S", 3300.0)),
            exec_timeout_s=max(10.0, _env_float("CAD_EXEC_TIMEOUT_S", 45.0)),
            llm_timeout_s=max(30.0, _env_float("CAD_LLM_TIMEOUT_S", 300.0)),
            max_tokens=max(2048, _env_int("CAD_MAX_TOKENS", 16384)),
            temperature=_env_float("CAD_TEMPERATURE", 0.8),
            top_p=_env_float("CAD_TOP_P", 0.95),
            seed=_env_int("CAD_SEED", 42),
            thinking_mode=os.environ.get("CAD_THINKING", "auto").strip().lower() or "auto",
            reasoning_effort=os.environ.get("CAD_REASONING_EFFORT", "auto").strip().lower() or "auto",
            iou_early_exit=_env_float("CAD_IOU_EARLY_EXIT", 0.95),
            vol_tol_pct=_env_float("CAD_VOL_TOL_PCT", 2.0),
            area_tol_pct=_env_float("CAD_AREA_TOL_PCT", 3.0),
            bbox_tol_pct=_env_float("CAD_BBOX_TOL_PCT", 1.5),
            max_turns_simple=max(1, _env_int("CAD_MAX_TURNS_SIMPLE", 4)),
            max_turns_medium=max(1, _env_int("CAD_MAX_TURNS_MEDIUM", 5)),
            max_turns_complex=max(1, _env_int("CAD_MAX_TURNS_COMPLEX", 6)),
        )
