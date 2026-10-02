"""Fan-out opportunity score (0-100) with transparent component breakdown."""

from __future__ import annotations

import math

import numpy as np

from ..analysis.embeddings import SemanticModel
from ..analysis.intent import BUSINESS_WEIGHT
from ..models import Coverage, DataLabel, Action, PageInventory, PageType, QueryCandidate
from ..research.serp_research import SerpResearch
from ..textutil import is_question

WEIGHTS = {
    "topical_relevance": 0.20,
    "intent_clarity": 0.15,
    "content_gap": 0.15,
    "search_volume": 0.15,
    "serp_evidence": 0.10,
    "source_frequency": 0.10,
    "business_relevance": 0.10,
    "ai_search_relevance": 0.05,
}

GAP_VALUE = {
    Coverage.CONTENT_GAP: 1.0,
    Coverage.PARTIALLY_COVERED: 0.7,
    Coverage.CANNIBALIZATION_RISK: 0.45,
    Coverage.EXISTING_PAGE: 0.35,
}
ORIGIN_WEIGHT = {"template": 0.25, "llm": 0.5}
SERP_ORIGINS = {"google", "paa", "related", "autocomplete"}
MONEY_TYPES = {PageType.PRODUCT, PageType.CATEGORY, PageType.COLLECTION, PageType.SERVICE, PageType.LOCATION, PageType.LANDING}


def compute_serp_evidence(cands: list[QueryCandidate], serp: SerpResearch, model: SemanticModel) -> None:
    if not serp.available:
        for c in cands:
            c.serp_evidence = 1.0 if c.origins & SERP_ORIGINS else 0.0
        return
    corpus = (
        [r.title for s in serp.serps for r in s.organic]
        + [e["question"] for e in serp.paa.values()]
        + list(serp.related)
    )
    if not corpus:
        return
    sim = SemanticModel.cosine_matrix(model.encode([c.query for c in cands]), model.encode(corpus))
    best = sim.max(axis=1)
    for c, s in zip(cands, best):
        c.serp_evidence = 1.0 if c.origins & SERP_ORIGINS else round(float(np.clip((s - 0.3) / 0.5, 0, 1)), 3)


def compute_business_relevance(cands: list[QueryCandidate], inventory: list[PageInventory], model: SemanticModel) -> None:
    money = [i for i in inventory if i.page_type in MONEY_TYPES]
    site_sim = np.zeros(len(cands))
    if money and cands:
        texts = [" . ".join([i.primary_topic] + i.products[:10] + i.services) for i in money]
        site_sim = SemanticModel.cosine_matrix(model.encode([c.query for c in cands]), model.encode(texts)).max(axis=1)
    for c, s in zip(cands, site_sim):
        base = BUSINESS_WEIGHT.get(c.intent, 0.5) if c.intent else 0.5
        c.business_relevance = round(float(0.65 * base + 0.35 * min(float(s) / 0.6, 1.0)), 3) if money else round(base, 3)


def compute_ai_relevance(c: QueryCandidate) -> float:
    words = c.query.split()
    score = 0.0
    if is_question(c.query):
        score += 0.45
    if len(words) >= 6:
        score += 0.2
    if "llm" in c.origins:
        score += 0.25
    if c.query_type in ("comparison", "how-to", "conversational question", "alternatives"):
        score += 0.1
    return round(min(score, 1.0), 3)


def score_candidates(cands: list[QueryCandidate]) -> None:
    volumes = [c.metrics.search_volume for c in cands if c.metrics and c.metrics.search_volume is not None]
    max_log = math.log10(max(volumes) + 1) if volumes else 0
    for c in cands:
        c.ai_search_relevance = compute_ai_relevance(c)
        weighted_origins = sum(ORIGIN_WEIGHT.get(o, 1.0) for o in c.origins)
        components = {
            "topical_relevance": c.topical_relevance,
            "intent_clarity": c.intent_confidence,
            "content_gap": GAP_VALUE.get(c.coverage, 0.0) if c.coverage else 0.0,  # type: ignore[arg-type]
            "serp_evidence": c.serp_evidence,
            "source_frequency": min(weighted_origins / 3.0, 1.0),
            "business_relevance": c.business_relevance,
            "ai_search_relevance": c.ai_search_relevance,
        }
        if c.action == Action.EXISTING_PAGE_OPTIMIZATION:
            components["content_gap"] = 0.35
        if c.metrics and c.metrics.search_volume is not None and max_log > 0:
            components["search_volume"] = math.log10(c.metrics.search_volume + 1) / max_log
        # Unknown volume is EXCLUDED (weights re-normalised) – never imputed.
        total_w = sum(WEIGHTS[k] for k in components)
        score = sum(WEIGHTS[k] * v for k, v in components.items()) / total_w
        c.score = round(100 * score, 1)
        c.score_breakdown = {k: round(v, 3) for k, v in components.items()}

        real_origins = [o for o in c.origins if o not in ORIGIN_WEIGHT]
        has_volume = "search_volume" in components
        if len(real_origins) >= 2 or (real_origins and has_volume):
            c.confidence = "HIGH"
        elif real_origins or has_volume:
            c.confidence = "MEDIUM"
        else:
            c.confidence = "LOW"
        c.label = DataLabel.REAL if real_origins else (DataLabel.LLM if "llm" in c.origins else DataLabel.TEMPLATE)
