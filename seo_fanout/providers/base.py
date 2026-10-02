"""Shared provider plumbing: errors, retries with exponential back-off, JSON parsing."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..logging_setup import get_logger

log = get_logger("providers")


class ProviderError(RuntimeError):
    """Non-retryable provider failure (bad request, auth, quota...)."""


class ProviderUnavailable(ProviderError):
    """Provider is not configured or its dependency is missing."""


class ProviderAuthError(ProviderError):
    pass


class RetryableProviderError(RuntimeError):
    """Transient failure (429 / 5xx / network) – retried with back-off."""


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning("Provider call failed (%s) – retry %d", exc, state.attempt_number)


api_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=2, max=30),
    retry=retry_if_exception_type((RetryableProviderError, httpx.TransportError)),
    before_sleep=_log_retry,
)


def check_response(resp: httpx.Response, provider: str) -> None:
    if resp.status_code in (401, 403):
        raise ProviderAuthError(f"{provider}: authentication failed (HTTP {resp.status_code})")
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RetryableProviderError(f"{provider}: HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise ProviderError(f"{provider}: HTTP {resp.status_code}: {resp.text[:300]}")


def extract_json(text: str) -> Any | None:
    """Parse JSON from an LLM answer that may include prose or code fences."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = cand.find(opener), cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start : end + 1])
                except json.JSONDecodeError:
                    continue
    return None
