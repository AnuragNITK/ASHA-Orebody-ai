"""Runtime configuration, loaded from environment / .env.

Every model id and endpoint is configurable so the pipeline is not pinned to a
model that may be renamed or retired on the provider side.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # optional: .env is a convenience, not a requirement
    from dotenv import load_dotenv

    # An explicit path, because find_dotenv() inspects the caller stack frame and
    # raises when the pipeline is driven from stdin, a notebook or an embedded host.
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass
OUTPUT_DIR = Path(os.getenv("MINEDESIGN_OUTPUT_DIR", PROJECT_ROOT / "outputs"))


def _f(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _i(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)


@dataclass
class Settings:
    # --- LLM access (Hugging Face Inference Providers, OpenAI-compatible router)
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    hf_base_url: str = field(
        default_factory=lambda: os.getenv(
            "HF_INFERENCE_BASE_URL", "https://router.huggingface.co/v1"
        )
    )
    model_orebody_grade: str = field(
        default_factory=lambda: os.getenv("MODEL_OREBODY_GRADE", "moonshotai/Kimi-K3")
    )
    model_pit_report: str = field(
        default_factory=lambda: os.getenv("MODEL_PIT_REPORT", "moonshotai/Kimi-K3")
    )

    # --- Local fallback (Ollama), only hit on validation failure
    local_fallback_model: str = field(
        default_factory=lambda: os.getenv("LOCAL_FALLBACK_MODEL", "qwen2.5:7b-instruct")
    )
    local_llm_base_url: str = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    )

    # --- LLM request behaviour
    llm_timeout_s: float = field(default_factory=lambda: _f("LLM_TIMEOUT_S", 180.0))
    llm_max_retries: int = field(default_factory=lambda: _i("LLM_MAX_RETRIES", 2))
    llm_temperature: float = field(default_factory=lambda: _f("LLM_TEMPERATURE", 0.2))
    llm_max_tokens: int = field(default_factory=lambda: _i("LLM_MAX_TOKENS", 8000))

    # --- Offline / deterministic mode: skip paid calls, use coded heuristics.
    offline: bool = field(
        default_factory=lambda: os.getenv("MINEDESIGN_OFFLINE", "0").lower()
        in ("1", "true", "yes")
    )

    # --- Engineering defaults (all overridable per run; reported as assumptions)
    default_density: float = field(default_factory=lambda: _f("DEFAULT_DENSITY", 2.7))

    @property
    def llm_available(self) -> bool:
        return bool(self.hf_token) and not self.offline


SETTINGS = Settings()
