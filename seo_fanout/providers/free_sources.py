"""Free data sources that need NO API key or account.

* Search-engine autocomplete (Google, Bing, DuckDuckGo) through the public
  OpenSearch suggestion endpoints that browsers use for their search box.
* Wikipedia (MediaWiki Action API) – subtopics and related entities.
* Reddit public search (``search.json``) – best effort; Reddit may rate-limit or
  block anonymous access, in which case the step is skipped.

All requests are rate-limited per host, cached in SQLite and fail softly.
Endpoint templates are class attributes so they can be pointed elsewhere (tests).
"""

from __future__ import annotations

import time
from urllib.parse import quote

import httpx

from ..db import Database
from ..logging_setup import get_logger
from ..models import CommunityItem

log = get_logger("free_sources")


class _PoliteClient:
    def __init__(self, user_agent: str, delay: float, timeout: float = 15.0):
        self.delay = delay
        self._last = 0.0
        self.calls = 0
        self.failures = 0
        self.client = httpx.Client(timeout=timeout, trust_env=True, follow_redirects=True,
                                   headers={"User-Agent": user_agent, "Accept": "application/json"})

    def get_json(self, url: str, params: dict | None = None):
        wait = self._last + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        for attempt in range(3):
            try:
                resp = self.client.get(url, params=params)
                self.calls += 1
                if resp.status_code == 429 or resp.status_code >= 500:
                    backoff = (2 ** attempt) * max(self.delay, 1.0) * 2
                    log.info("HTTP %s from %s – backing off %.0fs", resp.status_code, url.split("?")[0], backoff)
                    time.sleep(backoff)
                    continue
                if resp.status_code >= 400:
                    self.failures += 1
                    return None
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == 2:
                    self.failures += 1
                    log.info("Free source request failed (%s): %s", url.split("?")[0], type(exc).__name__)
                    return None
                time.sleep(2 ** attempt)
        self.failures += 1
        return None


# --------------------------------------------------------------------------- autocomplete
class AutocompleteEngine:
    """One search engine's public suggestion endpoint."""

    ENDPOINTS = {
        "google": "https://suggestqueries.google.com/complete/search",
        "bing": "https://api.bing.com/osjson.aspx",
        "duckduckgo": "https://duckduckgo.com/ac/",
    }

    def __init__(self, engine: str, db: Database, user_agent: str, delay: float = 1.0, ttl_days: int = 14):
        if engine not in self.ENDPOINTS:
            raise ValueError(f"Unknown autocomplete engine: {engine}")
        self.engine = engine
        self.db = db
        self.ttl_days = ttl_days
        self.http = _PoliteClient(user_agent, delay)
        self.consecutive_failures = 0

    @property
    def calls(self) -> int:
        return self.http.calls

    def _params(self, query: str, gl: str | None, hl: str) -> dict:
        if self.engine == "google":
            p = {"client": "firefox", "q": query, "hl": hl}
            if gl:
                p["gl"] = gl
            return p
        if self.engine == "bing":
            return {"query": query, "market": f"{hl}-{(gl or 'us').upper()}"}
        return {"q": query, "type": "list", "kl": f"{gl or 'wt'}-{hl}"}

    def suggest(self, query: str, gl: str | None, hl: str) -> list[str]:
        key = f"{self.engine}|{query}|{gl}|{hl}"
        cached = self.db.cache_get("suggest", key, self.ttl_days)
        if cached is not None:
            return cached
        if self.consecutive_failures >= 5:
            return []  # engine is blocking us – stop asking
        data = self.http.get_json(self.ENDPOINTS[self.engine], self._params(query, gl, hl))
        suggestions: list[str] = []
        # OpenSearch suggestion format: [query, [s1, s2, ...], ...]
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            suggestions = [s for s in data[1] if isinstance(s, str)]
        elif isinstance(data, list):  # DuckDuckGo legacy format: [{"phrase": ...}]
            suggestions = [d.get("phrase", "") for d in data if isinstance(d, dict) and d.get("phrase")]
        if data is None:
            self.consecutive_failures += 1
            return []
        self.consecutive_failures = 0
        self.db.cache_set("suggest", key, suggestions)
        return suggestions


# --------------------------------------------------------------------------- wikipedia
class WikipediaSource:
    API = "https://{lang}.wikipedia.org/w/api.php"

    def __init__(self, db: Database, user_agent: str, lang: str = "en", delay: float = 0.5):
        self.db = db
        self.lang = (lang or "en").split("-")[0]
        self.http = _PoliteClient(user_agent, delay)

    @property
    def calls(self) -> int:
        return self.http.calls

    def _api(self, params: dict):
        key = f"{self.lang}|{sorted(params.items())}"
        cached = self.db.cache_get("wikipedia", key, 30)
        if cached is not None:
            return cached
        data = self.http.get_json(self.API.format(lang=self.lang), {**params, "format": "json"})
        if data is not None:
            self.db.cache_set("wikipedia", key, data)
        return data

    def search_titles(self, query: str, limit: int = 8) -> list[str]:
        data = self._api({"action": "query", "list": "search", "srsearch": query, "srlimit": limit})
        try:
            return [r["title"] for r in data["query"]["search"]]
        except (TypeError, KeyError):
            return []

    def sections(self, title: str) -> list[str]:
        data = self._api({"action": "parse", "page": title, "prop": "sections", "redirects": 1})
        try:
            return [s["line"] for s in data["parse"]["sections"] if int(s.get("toclevel", 1)) <= 2]
        except (TypeError, KeyError, ValueError):
            return []


# --------------------------------------------------------------------------- reddit (public)
class RedditPublicSource:
    """Anonymous Reddit search. Best effort – Reddit may answer 403/429 to anonymous clients."""

    SEARCH = "https://www.reddit.com/search.json"
    name = "reddit_public"

    def __init__(self, db: Database, user_agent: str, delay: float = 2.0):
        self.db = db
        self.http = _PoliteClient(user_agent, delay)

    @property
    def calls(self) -> int:
        return self.http.calls

    def search(self, query: str, limit: int = 25) -> list[CommunityItem]:
        key = f"public|{query}|{limit}"
        data = self.db.cache_get("reddit", key, 14)
        if data is None:
            data = self.http.get_json(self.SEARCH, {"q": query, "limit": limit, "sort": "relevance", "t": "all"})
            if data is None:
                return []
            self.db.cache_set("reddit", key, data)
        items = []
        for child in (data.get("data", {}) or {}).get("children", []) if isinstance(data, dict) else []:
            d = child.get("data", {})
            if d.get("title"):
                items.append(CommunityItem(
                    source="reddit", platform_domain="reddit.com",
                    thread_url="https://www.reddit.com" + quote(d.get("permalink", ""), safe="/_-"),
                    title=d["title"], text=(d.get("selftext") or "")[:4000],
                    score=d.get("score"), num_comments=d.get("num_comments"), subreddit=d.get("subreddit"),
                ))
        return items

    def top_comments(self, permalink_url: str, limit: int = 20) -> list[str]:
        return []  # keep anonymous traffic minimal: titles + post bodies only
