"""Polite HTTP fetching with retries, back-off, per-host rate limiting and caching."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx

from .db import Database
from .logging_setup import get_logger

log = get_logger("http")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int | None
    headers: dict[str, str] = field(default_factory=dict)
    text: str = ""
    error: str | None = None
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").lower()

    @property
    def is_html(self) -> bool:
        ct = self.content_type
        return not ct or "html" in ct or "xhtml" in ct


def _retry_after_seconds(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except Exception:
            return default


class PoliteFetcher:
    """Synchronous fetcher that never hits the same host faster than ``delay``."""

    def __init__(
        self,
        user_agent: str,
        delay: float = 1.0,
        timeout: float = 20.0,
        max_retries: int = 3,
        db: Database | None = None,
        cache_ttl_days: float = 7,
        max_bytes: int = 5_000_000,
    ):
        self.delay = delay
        self.max_retries = max_retries
        self.db = db
        self.cache_ttl_days = cache_ttl_days
        self.max_bytes = max_bytes
        self.host_delays: dict[str, float] = {}
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()
        self.client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en;q=0.9,*;q=0.5",
            },
            timeout=timeout,
            follow_redirects=True,
            trust_env=True,
        )

    def close(self) -> None:
        self.client.close()

    def set_host_delay(self, host: str, delay: float) -> None:
        self.host_delays[host] = max(self.delay, delay)

    def _wait_turn(self, host: str) -> None:
        delay = self.host_delays.get(host, self.delay)
        with self._lock:
            last = self._last_request.get(host, 0.0)
            wait = last + delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request[host] = time.monotonic()

    def fetch(self, url: str, use_cache: bool = True) -> FetchResult:
        if use_cache and self.db is not None:
            cached = self.db.http_cache_get(url, self.cache_ttl_days)
            if cached is not None:
                return FetchResult(
                    url=url, final_url=cached["headers"].get("x-final-url", url),
                    status=cached["status"], headers=cached["headers"], text=cached["text"], from_cache=True,
                )

        host = urlsplit(url).netloc
        attempt = 0
        last_error: str | None = None
        while attempt <= self.max_retries:
            self._wait_turn(host)
            try:
                resp = self.client.get(url)
                headers = {k.lower(): v for k, v in resp.headers.items()}
                if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    backoff = (2 ** attempt) * max(self.delay, 1.0) + random.uniform(0, 0.5)
                    if resp.status_code == 429:
                        backoff = _retry_after_seconds(headers.get("retry-after"), backoff)
                        # A 429 means we are going too fast: slow down for the rest of the crawl.
                        self.host_delays[host] = min(self.host_delays.get(host, self.delay) * 2, 30.0)
                        log.warning("HTTP 429 from %s – backing off %.1fs and slowing crawl", host, backoff)
                    else:
                        log.info("HTTP %s for %s – retry %d in %.1fs", resp.status_code, url, attempt + 1, backoff)
                    time.sleep(min(backoff, 60))
                    attempt += 1
                    continue

                text = ""
                if len(resp.content) <= self.max_bytes:
                    text = resp.text
                else:
                    log.warning("Skipping body of %s (%d bytes > limit)", url, len(resp.content))
                headers["x-final-url"] = str(resp.url)
                result = FetchResult(url=url, final_url=str(resp.url), status=resp.status_code, headers=headers, text=text)
                if self.db is not None and resp.status_code < 500 and resp.status_code != 429:
                    self.db.http_cache_set(url, resp.status_code, headers, text)
                return result
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    backoff = (2 ** attempt) * max(self.delay, 1.0)
                    log.info("Network error for %s (%s) – retry %d in %.1fs", url, last_error, attempt + 1, backoff)
                    time.sleep(min(backoff, 30))
                attempt += 1
        log.warning("Failed to fetch %s: %s", url, last_error or "retries exhausted")
        return FetchResult(url=url, final_url=url, status=None, error=last_error or "retries exhausted")
