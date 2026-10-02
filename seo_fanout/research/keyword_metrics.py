"""Keyword metrics stage – real Google Ads data or explicit nulls (never estimates)."""

from __future__ import annotations

from dataclasses import dataclass

from ..logging_setup import get_logger
from ..models import DataLabel, KeywordMetrics, QueryCandidate
from ..providers.google_ads import GoogleAdsKeywordPlanner
from ..textutil import normalize_query

log = get_logger("keyword_metrics")


@dataclass
class MetricsResult:
    available: bool
    note: str
    requested: int = 0
    with_volume: int = 0


def attach_keyword_metrics(
    cands: list[QueryCandidate], planner: GoogleAdsKeywordPlanner | None, country: str, language: str,
    unavailable_reason: str,
) -> MetricsResult:
    if planner is None:
        for c in cands:
            c.metrics = KeywordMetrics(keyword=c.query, source="unavailable", label=DataLabel.UNAVAILABLE,
                                       note=unavailable_reason)
        return MetricsResult(available=False, note=unavailable_reason, requested=len(cands))
    try:
        found = planner.fetch([c.query for c in cands], country, language)
    except Exception as exc:
        reason = f"Google Ads request failed: {type(exc).__name__}: {str(exc)[:200]}"
        log.error(reason)
        for c in cands:
            c.metrics = KeywordMetrics(keyword=c.query, source="unavailable", label=DataLabel.UNAVAILABLE, note=reason)
        return MetricsResult(available=False, note=reason, requested=len(cands))
    with_volume = 0
    for c in cands:
        m = found.get(normalize_query(c.query))
        if m is None:
            c.metrics = KeywordMetrics(keyword=c.query, source="google_ads", label=DataLabel.UNAVAILABLE,
                                       note="Google Ads returned no data for this keyword")
        else:
            c.metrics = m
            if m.search_volume is not None:
                with_volume += 1
    return MetricsResult(available=True, note="Google Ads Keyword Planner", requested=len(cands), with_volume=with_volume)
