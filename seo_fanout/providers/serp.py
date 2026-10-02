"""SERP data providers (SerpApi, Serper.dev).

Google is never scraped directly: all SERP data comes from a licensed API.
Responses are cached in SQLite to avoid paying twice for the same query.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from ..config import Settings
from ..db import Database
from ..logging_setup import get_logger
from ..models import OrganicResult, SerpData
from .base import ProviderError, ProviderUnavailable, api_retry, check_response

log = get_logger("serp")


def _domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


class BaseSerpProvider:
    name = "base"

    def __init__(self, db: Database, ttl_days: int = 14):
        self.db = db
        self.ttl_days = ttl_days
        self.client = httpx.Client(timeout=40.0, trust_env=True)
        self.calls = 0

    def _search_raw(self, query: str, gl: str | None, hl: str, num: int, google_domain: str) -> dict:
        raise NotImplementedError

    def _autocomplete_raw(self, query: str, gl: str | None, hl: str) -> dict:
        raise NotImplementedError

    def _normalize(self, query: str, raw: dict) -> SerpData:
        raise NotImplementedError

    def _parse_autocomplete(self, raw: dict) -> list[str]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def search(self, query: str, gl: str | None, hl: str, num: int = 10, google_domain: str = "google.com") -> SerpData:
        key = f"{query}|{gl}|{hl}|{num}|{google_domain}"
        raw = self.db.cache_get(f"serp:{self.name}", key, self.ttl_days)
        if raw is None:
            try:
                raw = self._search_raw(query, gl, hl, num, google_domain)
                self.calls += 1
                self.db.cache_set(f"serp:{self.name}", key, raw)
                log.info("SERP API call (%s): %s", self.name, query)
            except ProviderError as exc:
                log.warning("SERP provider error for '%s': %s", query, exc)
                return SerpData(query=query, provider=self.name, error=str(exc))
            except Exception as exc:  # network failures after retries
                log.warning("SERP request failed for '%s': %s", query, exc)
                return SerpData(query=query, provider=self.name, error=f"{type(exc).__name__}: {exc}")
        return self._normalize(query, raw)

    def autocomplete(self, query: str, gl: str | None, hl: str) -> list[str]:
        key = f"ac|{query}|{gl}|{hl}"
        raw = self.db.cache_get(f"serp:{self.name}", key, self.ttl_days)
        if raw is None:
            try:
                raw = self._autocomplete_raw(query, gl, hl)
                self.calls += 1
                self.db.cache_set(f"serp:{self.name}", key, raw)
            except Exception as exc:
                log.info("Autocomplete unavailable for '%s': %s", query, exc)
                return []
        return self._parse_autocomplete(raw)


class SerpApiProvider(BaseSerpProvider):
    """https://serpapi.com – engine=google and engine=google_autocomplete."""

    name = "serpapi"
    endpoint = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, db: Database):
        super().__init__(db)
        self.api_key = api_key

    @api_retry
    def _get(self, params: dict) -> dict:
        resp = self.client.get(self.endpoint, params={**params, "api_key": self.api_key})
        check_response(resp, "SerpApi")
        data = resp.json()
        if data.get("error") and "hasn't returned any results" not in data["error"]:
            raise ProviderError(f"SerpApi: {data['error']}")
        return data

    def _search_raw(self, query, gl, hl, num, google_domain):
        params = {"engine": "google", "q": query, "hl": hl, "num": num, "google_domain": google_domain}
        if gl:
            params["gl"] = gl
        return self._get(params)

    def _autocomplete_raw(self, query, gl, hl):
        params = {"engine": "google_autocomplete", "q": query, "hl": hl}
        if gl:
            params["gl"] = gl
        return self._get(params)

    def _parse_autocomplete(self, raw):
        return [s.get("value", "") for s in raw.get("suggestions", []) if s.get("value")]

    def _normalize(self, query, raw):
        organic = [
            OrganicResult(
                position=int(r.get("position") or i + 1), url=r.get("link", ""), title=r.get("title", ""),
                snippet=r.get("snippet", ""), domain=_domain(r.get("link", "")),
            )
            for i, r in enumerate(raw.get("organic_results", []) or [])
            if r.get("link")
        ]
        paa = [
            {"question": q.get("question", ""), "snippet": q.get("snippet", "") or "", "link": q.get("link", "") or ""}
            for q in raw.get("related_questions", []) or []
            if q.get("question")
        ]
        related = [r.get("query", "") for r in raw.get("related_searches", []) or [] if r.get("query")]
        fs = None
        ab = raw.get("answer_box")
        if isinstance(ab, dict) and (ab.get("snippet") or ab.get("answer")):
            fs = {"title": ab.get("title", ""), "snippet": ab.get("snippet") or ab.get("answer", ""), "link": ab.get("link", "")}
        entities = []
        kg = raw.get("knowledge_graph")
        if isinstance(kg, dict) and kg.get("title"):
            entities.append(kg["title"])
            for k in ("type",):
                if kg.get(k):
                    entities.append(str(kg[k]))
        return SerpData(query=query, provider=self.name, organic=organic, paa=paa, related_searches=related,
                        featured_snippet=fs, entities=entities)


class SerperProvider(BaseSerpProvider):
    """https://serper.dev – Google SERP API."""

    name = "serper"

    def __init__(self, api_key: str, db: Database):
        super().__init__(db)
        self.api_key = api_key

    @api_retry
    def _post(self, endpoint: str, payload: dict) -> dict:
        resp = self.client.post(
            f"https://google.serper.dev/{endpoint}", json=payload,
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
        )
        check_response(resp, "Serper")
        return resp.json()

    def _search_raw(self, query, gl, hl, num, google_domain):
        payload = {"q": query, "hl": hl, "num": num}
        if gl:
            payload["gl"] = gl
        return self._post("search", payload)

    def _autocomplete_raw(self, query, gl, hl):
        payload = {"q": query, "hl": hl}
        if gl:
            payload["gl"] = gl
        return self._post("autocomplete", payload)

    def _parse_autocomplete(self, raw):
        return [s.get("value", "") for s in raw.get("suggestions", []) if s.get("value")]

    def _normalize(self, query, raw):
        organic = [
            OrganicResult(
                position=int(r.get("position") or i + 1), url=r.get("link", ""), title=r.get("title", ""),
                snippet=r.get("snippet", ""), domain=_domain(r.get("link", "")),
            )
            for i, r in enumerate(raw.get("organic", []) or [])
            if r.get("link")
        ]
        paa = [
            {"question": q.get("question", ""), "snippet": q.get("snippet", "") or "", "link": q.get("link", "") or ""}
            for q in raw.get("peopleAlsoAsk", []) or []
            if q.get("question")
        ]
        related = [r.get("query", "") for r in raw.get("relatedSearches", []) or [] if r.get("query")]
        fs = None
        ab = raw.get("answerBox")
        if isinstance(ab, dict) and (ab.get("snippet") or ab.get("answer")):
            fs = {"title": ab.get("title", ""), "snippet": ab.get("snippet") or ab.get("answer", ""), "link": ab.get("link", "")}
        entities = []
        kg = raw.get("knowledgeGraph")
        if isinstance(kg, dict) and kg.get("title"):
            entities.append(kg["title"])
            if kg.get("type"):
                entities.append(str(kg["type"]))
        return SerpData(query=query, provider=self.name, organic=organic, paa=paa, related_searches=related,
                        featured_snippet=fs, entities=entities)


def get_serp_provider(settings: Settings, db: Database) -> BaseSerpProvider:
    if settings.serp_provider == "serper":
        if not settings.serper_api_key:
            raise ProviderUnavailable("SERPER_API_KEY not set")
        return SerperProvider(settings.serper_api_key, db)
    if not settings.serp_api_key:
        raise ProviderUnavailable("SERP_API_KEY not set")
    return SerpApiProvider(settings.serp_api_key, db)
