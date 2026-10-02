"""Google SERP research through a licensed SERP API."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..db import Database
from ..logging_setup import get_logger
from ..models import SerpData
from ..providers.locale import country_info, language_info
from ..providers.serp import BaseSerpProvider
from ..textutil import content_tokens, normalize_query

log = get_logger("serp_research")

PATTERNS = [
    "{seed}",
    "best {seed}",
    "how to {seed}",
    "why {seed}",
    "{seed} for",
    "{seed} vs",
    "{seed} alternatives",
    "{seed} benefits",
    "{seed} problems",
    "{seed} price",
    "{seed} reviews",
    "{seed} in {country}",
    "what is {seed}",
    "{seed} side effects",
]

AUTOCOMPLETE_PATTERNS = ["{seed}", "{seed} for", "{seed} vs", "how to {seed}", "best {seed}", "why {seed}"]


@dataclass
class SerpResearch:
    available: bool
    provider: str = ""
    note: str = ""
    serps: list[SerpData] = field(default_factory=list)
    autocomplete: dict[str, list[str]] = field(default_factory=dict)
    related: Counter = field(default_factory=Counter)
    modifiers: Counter = field(default_factory=Counter)
    entities: Counter = field(default_factory=Counter)
    ranking_domains: Counter = field(default_factory=Counter)
    paa: dict[str, dict] = field(default_factory=dict)  # normalized question -> {question, count, source_queries, snippet, link}

    def all_titles(self) -> list[str]:
        return [r.title for s in self.serps for r in s.organic]

    def add_paa(self, serp: SerpData, depth: int = 1) -> None:
        for item in serp.paa:
            q = item["question"].strip()
            key = normalize_query(q)
            if not key:
                continue
            entry = self.paa.setdefault(key, {"question": q, "count": 0, "source_queries": [], "snippet": item.get("snippet", ""),
                                              "link": item.get("link", ""), "depth": depth})
            entry["count"] += 1
            if serp.query not in entry["source_queries"]:
                entry["source_queries"].append(serp.query)


def build_serp_queries(seed: str, country: str, limit: int) -> list[str]:
    queries = []
    for pattern in PATTERNS:
        q = pattern.format(seed=seed, country=country or "").strip()
        # "how to sunscreen for oily skin" is awkward; keep it – SERP APIs handle it, and Google
        # rewrites it, which itself reveals the dominant how-to needs.
        if q not in queries:
            queries.append(q)
    return queries[:limit]


def run_serp_research(
    provider: BaseSerpProvider | None, db: Database, run_id: str, seed: str, country: str, language: str, limit: int, num: int = 10
) -> SerpResearch:
    if provider is None:
        return SerpResearch(available=False, note="No SERP provider configured (set SERP_API_KEY or SERPER_API_KEY). "
                                                  "SERP data is NOT fabricated – related columns stay empty.")
    gl, _, google_domain = country_info(country)
    hl, _ = language_info(language)
    research = SerpResearch(available=True, provider=provider.name)
    seed_tokens = set(content_tokens(seed))

    for q in build_serp_queries(seed, country, limit):
        serp = provider.search(q, gl, hl, num=num, google_domain=google_domain)
        if serp.error:
            if not research.serps and "auth" in serp.error.lower():
                research.available = False
                research.note = f"SERP provider authentication failed: {serp.error}"
                log.error(research.note)
                return research
            continue
        research.serps.append(serp)
        research.add_paa(serp)
        for rel in serp.related_searches:
            research.related[rel.strip().lower()] += 1
        for ent in serp.entities:
            research.entities[ent] += 1
        for r in serp.organic:
            research.ranking_domains[r.domain] += 1
        db.executemany(
            "INSERT INTO serp_results(run_id,query,provider,position,url,title,snippet,result_type) VALUES(?,?,?,?,?,?,?,?)",
            [(run_id, q, serp.provider, r.position, r.url, r.title, r.snippet, "organic") for r in serp.organic]
            + [(run_id, q, serp.provider, None, rel, rel, "", "related_search") for rel in serp.related_searches]
            + ([(run_id, q, serp.provider, 0, serp.featured_snippet.get("link", ""), serp.featured_snippet.get("title", ""),
                 serp.featured_snippet.get("snippet", ""), "featured_snippet")] if serp.featured_snippet else []),
        )

    for pattern in AUTOCOMPLETE_PATTERNS:
        q = pattern.format(seed=seed)
        suggestions = provider.autocomplete(q, gl, hl)
        if suggestions:
            research.autocomplete[q] = suggestions

    # Modifiers: words that co-occur with the seed in related searches / autocomplete.
    for phrase in list(research.related) + [s for v in research.autocomplete.values() for s in v]:
        extra = [t for t in content_tokens(phrase) if t not in seed_tokens]
        for t in extra:
            research.modifiers[t] += 1
    log.info(
        "SERP research: %d SERPs, %d PAA, %d related, %d autocomplete groups",
        len(research.serps), len(research.paa), len(research.related), len(research.autocomplete),
    )
    if not research.serps:
        research.available = False
        research.note = "SERP provider returned no usable results (see logs/research.log)."
    return research
