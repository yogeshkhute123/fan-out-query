"""OpenAI provider (Chat Completions REST API)."""

from __future__ import annotations

import httpx

from ..db import Database
from .base import ProviderError, ProviderUnavailable, api_retry, check_response
from .llm import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    name = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str | None, model: str, db: Database):
        if not api_key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
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
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        check_response(resp, "OpenAI")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"OpenAI: unexpected response shape: {exc}") from exc
        return LLMResponse(text=text)
