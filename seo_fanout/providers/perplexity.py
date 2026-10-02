"""Perplexity provider (Sonar chat completions API – web-grounded answers with citations)."""

from __future__ import annotations

import httpx

from ..db import Database
from .base import ProviderError, ProviderUnavailable, api_retry, check_response
from .llm import LLMProvider, LLMResponse


class PerplexityProvider(LLMProvider):
    name = "perplexity"
    endpoint = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str | None, model: str, db: Database):
        if not api_key:
            raise ProviderUnavailable("PERPLEXITY_API_KEY not set")
        super().__init__(model, db)
        self.client = httpx.Client(
            timeout=120.0, trust_env=True,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @api_retry
    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        resp = self.client.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
            },
        )
        check_response(resp, "Perplexity")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Perplexity: unexpected response shape: {exc}") from exc
        citations = list(data.get("citations") or [])
        for result in data.get("search_results") or []:
            if isinstance(result, dict) and result.get("url") and result["url"] not in citations:
                citations.append(result["url"])
        return LLMResponse(text=text, citations=citations)
