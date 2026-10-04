"""LLM transport: the paid HF router, plus the free local fallback.

Two paid calls are made per pipeline run and no more. Everything here is
plumbing -- retries, JSON extraction, and a layered fallback so that a missing
token, an exhausted quota or an unreachable endpoint degrades the run instead of
ending it. Which path actually ran is recorded and stated in the report, because
a report that quietly swapped an engineering judgement for a default is worse
than one that says so.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import SETTINGS
from ..errors import LLMError
from ..logging_utils import get_logger

log = get_logger(__name__)

# Status codes where retrying cannot help: the request is refused, not flaky.
_FATAL_STATUS = {400, 401, 402, 403, 404, 422}


class LLMUnavailable(LLMError):
    """The model could not be reached or is not permitted for this account."""


@dataclass
class LLMCallRecord:
    """Provenance for one call, surfaced in the report."""

    label: str
    provider: str            # "hf" | "ollama" | "deterministic"
    model: str
    ok: bool
    duration_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    attempts: int = 1

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CallLog:
    """Collects every LLM interaction in a run."""

    records: list[LLMCallRecord] = field(default_factory=list)

    def add(self, record: LLMCallRecord) -> LLMCallRecord:
        self.records.append(record)
        return record

    def as_list(self) -> list[dict]:
        return [r.as_dict() for r in self.records]

    @property
    def paid_calls(self) -> int:
        return sum(1 for r in self.records if r.provider == "hf" and r.ok)


CALL_LOG = CallLog()


# ---------------------------------------------------------------- JSON parsing
def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that a bare json.loads is
    not a reliable contract, so this falls back to fence stripping and then to
    brace matching before giving up.
    """
    if not text or not text.strip():
        raise LLMError("model returned an empty response")

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise LLMError(f"could not parse JSON from model response: {text[:400]}")


# ------------------------------------------------------------------ transports
def _call_hf(system: str, user: str, model: str, *, json_mode: bool,
             max_tokens: int, temperature: float) -> tuple[str, dict]:
    if not SETTINGS.hf_token:
        raise LLMUnavailable("HF_TOKEN is not set")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    url = f"{SETTINGS.hf_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {SETTINGS.hf_token}",
               "Content-Type": "application/json"}

    with httpx.Client(timeout=SETTINGS.llm_timeout_s) as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code in _FATAL_STATUS:
        detail = response.text[:300]
        raise LLMUnavailable(f"HF router returned {response.status_code}: {detail}")
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return content, usage


def _call_ollama(system: str, user: str, model: str, *, json_mode: bool,
                 max_tokens: int, temperature: float) -> tuple[str, dict]:
    url = f"{SETTINGS.local_llm_base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        with httpx.Client(timeout=SETTINGS.llm_timeout_s) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"local fallback model at {SETTINGS.local_llm_base_url} "
                             f"is not reachable: {exc}") from exc

    if response.status_code == 404:
        raise LLMUnavailable(f"local model {model} is not pulled; run: ollama pull {model}")
    response.raise_for_status()

    data = response.json()
    content = data.get("message", {}).get("content", "")
    usage = {"prompt_tokens": data.get("prompt_eval_count"),
             "completion_tokens": data.get("eval_count")}
    return content, usage


# --------------------------------------------------------------- public façade
def chat_json(system: str, user: str, *, model: str | None = None, label: str = "llm",
              provider: str = "hf", json_mode: bool = True,
              max_tokens: int | None = None, temperature: float | None = None,
              call_log: CallLog | None = None) -> dict:
    """One chat completion returning parsed JSON, with retries and provenance."""
    log_target = call_log or CALL_LOG
    model = model or (SETTINGS.model_orebody_grade if provider == "hf"
                      else SETTINGS.local_fallback_model)
    max_tokens = max_tokens or SETTINGS.llm_max_tokens
    temperature = SETTINGS.llm_temperature if temperature is None else temperature
    transport = _call_hf if provider == "hf" else _call_ollama

    if provider == "hf" and SETTINGS.offline:
        raise LLMUnavailable("MINEDESIGN_OFFLINE is set; paid calls are disabled")

    started = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(1, SETTINGS.llm_max_retries + 2):
        try:
            content, usage = transport(system, user, model, json_mode=json_mode,
                                       max_tokens=max_tokens, temperature=temperature)
            parsed = extract_json(content)
            log_target.add(LLMCallRecord(
                label=label, provider=provider, model=model, ok=True,
                duration_s=time.perf_counter() - started,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                attempts=attempt))
            log.info("%s: %s/%s responded in %.1fs", label, provider, model,
                     time.perf_counter() - started)
            return parsed
        except LLMUnavailable as exc:
            last_error = exc
            break  # refusal, not flakiness
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt <= SETTINGS.llm_max_retries:
                backoff = 2.0 ** attempt
                log.warning("%s attempt %d failed (%s); retrying in %.0fs",
                            label, attempt, exc, backoff)
                time.sleep(backoff)

    log_target.add(LLMCallRecord(
        label=label, provider=provider, model=model, ok=False,
        duration_s=time.perf_counter() - started, error=str(last_error),
        attempts=SETTINGS.llm_max_retries + 1))
    raise LLMError(f"{label} failed via {provider}/{model}: {last_error}") from last_error


def chat_json_with_fallback(system: str, user: str, *, label: str,
                            model: str | None = None,
                            max_tokens: int | None = None,
                            call_log: CallLog | None = None) -> tuple[dict | None, str]:
    """Try the paid router, then the local model. Returns ``(result, provider)``.

    ``result`` is None when every path failed, which the caller answers with its
    own deterministic heuristic rather than by inventing numbers.
    """
    try:
        return chat_json(system, user, model=model, label=label, provider="hf",
                         max_tokens=max_tokens, call_log=call_log), "hf"
    except LLMError as exc:
        log.warning("%s: paid call unavailable (%s); trying the local fallback model",
                    label, exc)

    try:
        return chat_json(system, user, model=SETTINGS.local_fallback_model,
                         label=f"{label}-local", provider="ollama",
                         max_tokens=max_tokens, call_log=call_log), "ollama"
    except LLMError as exc:
        log.warning("%s: local fallback also unavailable (%s); "
                    "falling back to the deterministic heuristic", label, exc)

    return None, "deterministic"


# ------------------------------------------------- ingestion column-map rescue
_COLUMN_MAP_SYSTEM = """You map CSV column names onto a fixed schema for mining drillhole data.
Answer with strict JSON only: {"mapping": {"<source column name>": "<canonical name>"}}
Use only canonical names from the list you are given. Omit any source column you
cannot confidently map. Never invent a column name that is not in the source list."""


def map_columns_with_fallback_model(file_kind: str, columns: list[str],
                                    unresolved: list[str],
                                    sample_rows: list[dict]) -> dict[str, str]:
    """Local-model rescue for one file whose columns the dictionary could not map."""
    user = json.dumps({
        "file_kind": file_kind,
        "source_columns": columns,
        "canonical_names_needed": unresolved,
        "sample_rows": sample_rows,
    }, indent=2, default=str)

    result = chat_json(_COLUMN_MAP_SYSTEM, user, model=SETTINGS.local_fallback_model,
                       label=f"column-map-{file_kind}", provider="ollama",
                       max_tokens=800)
    mapping = result.get("mapping", {})
    if not isinstance(mapping, dict):
        raise LLMError("column mapper returned a mapping that is not an object")
    return {str(k): str(v) for k, v in mapping.items()}
