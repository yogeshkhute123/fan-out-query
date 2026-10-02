"""Anthropic (Claude) provider using the official ``anthropic`` Python SDK."""

from __future__ import annotations

from ..db import Database
from ..logging_setup import get_logger
from .base import ProviderAuthError, ProviderError, ProviderUnavailable, RetryableProviderError
from .llm import LLMProvider, LLMResponse

log = get_logger("anthropic")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None, model: str, db: Database):
        if not api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ProviderUnavailable("anthropic package not installed (pip install anthropic)") from exc
        super().__init__(model, db)
        self._sdk = anthropic
        # The SDK retries 408/409/429/5xx and connection errors with exponential back-off.
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=4, timeout=300.0)

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        sdk = self._sdk
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max(max_tokens, 16000),
                system=system,
                messages=[{"role": "user", "content": user}],
                # Server-side refusal fallback + explicit effort level.
                extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
                extra_body={"fallbacks": "default", "output_config": {"effort": "medium"}},
            )
        except sdk.AuthenticationError as exc:
            raise ProviderAuthError("Anthropic: authentication failed") from exc
        except sdk.RateLimitError as exc:
            raise RetryableProviderError("Anthropic: rate limited after SDK retries") from exc
        except sdk.BadRequestError as exc:
            raise ProviderError(f"Anthropic: bad request: {exc.message}") from exc
        except sdk.APIStatusError as exc:
            raise ProviderError(f"Anthropic: HTTP {exc.status_code}") from exc
        except sdk.APIConnectionError as exc:
            raise ProviderError("Anthropic: connection error") from exc

        if response.stop_reason == "refusal":
            raise ProviderError("Anthropic: request declined (refusal)")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(text=text)
