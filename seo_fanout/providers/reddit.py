"""Reddit provider using the official OAuth API (application-only, read-only)."""

from __future__ import annotations

import time

import httpx

from ..config import Settings
from ..db import Database
from ..logging_setup import get_logger
from ..models import CommunityItem
from .base import ProviderUnavailable, api_retry, check_response

log = get_logger("reddit")


class RedditProvider:
    name = "reddit"
    token_url = "https://www.reddit.com/api/v1/access_token"
    api = "https://oauth.reddit.com"

    def __init__(self, settings: Settings, db: Database):
        if not settings.reddit_configured:
            raise ProviderUnavailable("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
        self.settings = settings
        self.db = db
        self.client = httpx.Client(timeout=30.0, trust_env=True, headers={"User-Agent": settings.reddit_user_agent})
        self._token: str | None = None
        self._token_expiry = 0.0
        self._last_call = 0.0
        self.calls = 0

    def _auth(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        resp = self.client.post(
            self.token_url,
            data={"grant_type": "client_credentials"},
            auth=(self.settings.reddit_client_id or "", self.settings.reddit_client_secret or ""),
        )
        check_response(resp, "Reddit auth")
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + float(data.get("expires_in", 3600))
        return self._token

    @api_retry
    def _get(self, path: str, params: dict) -> dict | list:
        # Stay well under Reddit's 100 requests/minute limit.
        wait = self._last_call + 1.0 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        resp = self.client.get(f"{self.api}{path}", params=params, headers={"Authorization": f"bearer {self._auth()}"})
        check_response(resp, "Reddit")
        self.calls += 1
        return resp.json()

    def search(self, query: str, limit: int = 25) -> list[CommunityItem]:
        key = f"search|{query}|{limit}"
        cached = self.db.cache_get("reddit", key, 14)
        if cached is None:
            cached = self._get("/search", {"q": query, "limit": limit, "sort": "relevance", "t": "all", "type": "link"})
            self.db.cache_set("reddit", key, cached)
        items = []
        for child in (cached.get("data", {}) or {}).get("children", []):  # type: ignore[union-attr]
            d = child.get("data", {})
            if not d.get("title"):
                continue
            items.append(
                CommunityItem(
                    source="reddit", platform_domain="reddit.com",
                    thread_url="https://www.reddit.com" + d.get("permalink", ""),
                    title=d.get("title", ""), text=(d.get("selftext") or "")[:4000],
                    score=d.get("score"), num_comments=d.get("num_comments"), subreddit=d.get("subreddit"),
                )
            )
        return items

    def top_comments(self, permalink_url: str, limit: int = 20) -> list[str]:
        path = permalink_url.replace("https://www.reddit.com", "").rstrip("/")
        key = f"comments|{path}|{limit}"
        cached = self.db.cache_get("reddit", key, 14)
        if cached is None:
            cached = self._get(f"{path}.json", {"limit": limit, "depth": 1, "sort": "top"})
            self.db.cache_set("reddit", key, cached)
        comments: list[str] = []
        if isinstance(cached, list) and len(cached) > 1:
            for child in cached[1].get("data", {}).get("children", []):
                body = child.get("data", {}).get("body")
                if body:
                    comments.append(body[:2000])
        return comments
