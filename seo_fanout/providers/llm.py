"""Common interface for LLM providers, with response caching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..db import Database
from ..logging_setup import get_logger
from .base import ProviderError, extract_json

log = get_logger("llm")


@dataclass
class LLMResponse:
    text: str
    citations: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""


class LLMProvider:
    name = "llm"

    def __init__(self, model: str, db: Database, ttl_days: int = 30):
        self.model = model
        self.db = db
        self.ttl_days = ttl_days
        self.calls = 0
        self.failures = 0

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        raise NotImplementedError

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> LLMResponse | None:
        key = f"{self.model}|{system}|{user}|{max_tokens}"
        cached = self.db.cache_get(f"llm:{self.name}", key, self.ttl_days)
        if cached is not None:
            return LLMResponse(**cached)
        try:
            resp = self._complete(system, user, max_tokens)
            self.calls += 1
        except ProviderError as exc:
            self.failures += 1
            log.warning("%s error: %s", self.name, exc)
            return None
        except Exception as exc:  # network/timeouts after retries, SDK errors
            self.failures += 1
            log.warning("%s request failed: %s: %s", self.name, type(exc).__name__, str(exc)[:300])
            return None
        resp.provider, resp.model = self.name, self.model
        self.db.cache_set(f"llm:{self.name}", key, resp.__dict__)
        return resp

    def complete_json(self, system: str, user: str, max_tokens: int = 4000) -> tuple[Any | None, LLMResponse | None]:
        resp = self.complete(system, user, max_tokens)
        if resp is None:
            return None, None
        data = extract_json(resp.text)
        if data is None:
            log.warning("%s returned non-JSON output; ignoring", self.name)
        return data, resp
