"""People-Also-Ask research: aggregation plus a bounded second-level expansion."""

from __future__ import annotations

from ..db import Database
from ..logging_setup import get_logger
from ..providers.locale import country_info, language_info
from ..providers.serp import BaseSerpProvider
from .serp_research import SerpResearch

log = get_logger("paa")


def expand_paa(
    provider: BaseSerpProvider | None, research: SerpResearch, db: Database, run_id: str,
    country: str, language: str, max_expansions: int = 4,
) -> int:
    """Search the most frequent PAA questions to surface follow-up questions.

    Returns the number of PAA questions after expansion.
    """
    if provider is None or not research.available:
        return len(research.paa)
    gl, _, google_domain = country_info(country)
    hl, _ = language_info(language)
    top = sorted(research.paa.values(), key=lambda e: -e["count"])[:max_expansions]
    for entry in top:
        serp = provider.search(entry["question"], gl, hl, num=10, google_domain=google_domain)
        if serp.error:
            continue
        research.serps.append(serp)
        research.add_paa(serp, depth=2)
        for rel in serp.related_searches:
            research.related[rel.strip().lower()] += 1

    db.executemany(
        "INSERT INTO paa_questions(run_id,source_query,question,snippet,link) VALUES(?,?,?,?,?)",
        [(run_id, "; ".join(e["source_queries"]), e["question"], e.get("snippet", ""), e.get("link", ""))
         for e in research.paa.values()],
    )
    log.info("PAA questions after expansion: %d", len(research.paa))
    return len(research.paa)
