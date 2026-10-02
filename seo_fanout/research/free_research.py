"""Key-free demand research: autocomplete expansion and Wikipedia subtopics.

Autocomplete suggestions are *real* queries that search engines have observed
often enough to suggest. Agreement between engines and a high suggestion rank
are used as an explicitly ESTIMATED demand signal – never as search volume.
"""

from __future__ import annotations

import string
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..logging_setup import get_logger
from ..providers.free_sources import AutocompleteEngine, WikipediaSource
from ..providers.locale import country_info, language_info
from ..textutil import normalize_query, stem_tokens
from .serp_research import SerpResearch

log = get_logger("free_research")

QUESTION_PREFIXES = ["what", "why", "how", "how to", "when", "where", "who", "which", "can", "does", "is", "are",
                     "should", "will"]
PREPOSITION_SUFFIXES = ["for", "with", "without", "vs", "or", "and", "near", "to", "like", "in", "after", "before"]
COMMERCIAL_PATTERNS = ["best {s}", "{s} price", "buy {s}", "cheap {s}", "{s} review", "top {s}", "{s} brands"]
GENERIC_SECTIONS = {
    "see also", "references", "external links", "notes", "further reading", "bibliography", "gallery", "sources",
    "citations", "footnotes", "etymology", "overview", "contents", "notes and references",
}


@dataclass
class FreeResearchStats:
    engines_used: list[str] = field(default_factory=list)
    engines_failed: list[str] = field(default_factory=list)
    probes: int = 0
    suggestions: int = 0
    wikipedia_title: str | None = None
    wikipedia_sections: int = 0
    calls: dict[str, int] = field(default_factory=dict)


def build_probes(seed: str, depth: str = "basic") -> list[str]:
    s = normalize_query(seed)
    probes = [s]
    probes += [f"{q} {s}" for q in QUESTION_PREFIXES]
    probes += [f"{s} {p}" for p in PREPOSITION_SUFFIXES]
    probes += [p.format(s=s) for p in COMMERCIAL_PATTERNS]
    if depth == "full":
        probes += [f"{s} {letter}" for letter in string.ascii_lowercase]
    return list(dict.fromkeys(probes))


def _run_engine(engine: AutocompleteEngine, probes: list[str], gl: str | None, hl: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for probe in probes:
        suggestions = engine.suggest(probe, gl, hl)
        if suggestions:
            out[probe] = suggestions
        if engine.consecutive_failures >= 5:
            log.warning("Autocomplete engine '%s' keeps failing – skipping the rest of its probes", engine.engine)
            break
    return out


def run_autocomplete_research(
    research: SerpResearch, engines: list[AutocompleteEngine], seed: str, country: str, language: str,
    depth: str = "basic",
) -> FreeResearchStats:
    """Fill ``research.autocomplete`` and ``research.suggest_meta`` from all engines (in parallel, one thread per host)."""
    stats = FreeResearchStats()
    if not engines:
        return stats
    gl, _, _ = country_info(country)
    hl, _ = language_info(language)
    probes = build_probes(seed, depth)
    stats.probes = len(probes)
    seed_stems = set(stem_tokens(seed))
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        results = list(pool.map(lambda e: (e, _run_engine(e, probes, gl, hl)), engines))
    for engine, by_probe in results:
        stats.calls[engine.engine] = engine.calls
        if not by_probe:
            stats.engines_failed.append(engine.engine)
            continue
        stats.engines_used.append(engine.engine)
        for probe, suggestions in by_probe.items():
            research.autocomplete[f"{engine.engine}: {probe}"] = suggestions
            for rank, sug in enumerate(suggestions, start=1):
                norm = normalize_query(sug)
                if not norm or not (set(stem_tokens(norm)) & seed_stems):
                    continue  # drifted away from the topic
                meta = research.suggest_meta.setdefault(
                    norm, {"text": sug.strip().lower(), "engines": set(), "best_rank": rank, "probes": []}
                )
                meta["engines"].add(engine.engine)
                meta["best_rank"] = min(meta["best_rank"], rank)
                if probe not in meta["probes"] and len(meta["probes"]) < 5:
                    meta["probes"].append(probe)
    stats.suggestions = len(research.suggest_meta)
    log.info("Autocomplete research: %d probes, %d distinct on-topic suggestions from %s (failed: %s)",
             stats.probes, stats.suggestions, ", ".join(stats.engines_used) or "none",
             ", ".join(stats.engines_failed) or "none")
    return stats


def run_wikipedia_research(research: SerpResearch, wiki: WikipediaSource | None, head: str,
                           stats: FreeResearchStats) -> None:
    if wiki is None:
        return
    titles = wiki.search_titles(head)
    stats.calls["wikipedia"] = wiki.calls
    if not titles:
        return
    main = titles[0]
    sections = [s for s in wiki.sections(main) if s.strip().lower() not in GENERIC_SECTIONS]
    stats.wikipedia_title = main
    stats.wikipedia_sections = len(sections)
    stats.calls["wikipedia"] = wiki.calls
    research.wikipedia = {"title": main, "sections": sections, "related": titles[1:]}
    for t in titles[1:]:
        research.entities[t] += 1
    log.info("Wikipedia: '%s' with %d sections, %d related articles", main, len(sections), len(titles) - 1)


def demand_proxy(meta: dict | None, n_engines: int) -> float | None:
    """ESTIMATED demand signal in [0,1] from engine agreement and suggestion rank (not search volume)."""
    if not meta or n_engines == 0:
        return None
    agreement = len(meta["engines"]) / n_engines
    rank_score = max(0.0, 1.0 - (meta["best_rank"] - 1) / 10)
    return round(0.6 * agreement + 0.4 * rank_score, 3)
