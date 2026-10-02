"""Quality control: relevance scoring and rejection rules applied before export."""

from __future__ import annotations

import re

import numpy as np

from ..analysis.embeddings import SemanticModel
from ..logging_setup import get_logger
from ..models import Action, QueryCandidate
from ..textutil import STOPWORDS, normalize_query, stem_tokens, tokens

log = get_logger("quality")

JUNK = re.compile(r"(https?://|www\.|\.com\b|@|#|\b(lol|lmao|upvote|downvote|edit:|tl;?dr|reddit|quora)\b)", re.I)
GENERIC_ONLY = {"guide", "tips", "info", "information", "help", "question", "questions", "thread", "discussion", "update"}


def compute_topical_relevance(cands: list[QueryCandidate], seed: str, model: SemanticModel) -> None:
    if not cands:
        return
    emb = model.encode([c.query for c in cands])
    seed_emb = model.encode([seed])
    sims = (emb @ seed_emb.T).ravel()
    seed_stems = set(stem_tokens(seed))
    for c, s in zip(cands, sims):
        stems = set(stem_tokens(c.query))
        overlap = len(stems & seed_stems) / max(len(seed_stems), 1)
        c.topical_relevance = round(float(np.clip(0.55 * s + 0.45 * overlap, 0, 1)), 3)


def reject_reason(c: QueryCandidate, seed: str, min_relevance: float = 0.22) -> str | None:
    q = normalize_query(c.query)
    words = q.split()
    content = [w for w in words if w not in STOPWORDS]
    if len(words) < 2 or not content:
        return "meaningless: too short / no content words"
    if len(words) > 18:
        return "meaningless: too long to be a search query"
    if JUNK.search(c.query):
        return "meaningless: contains URL/forum artefacts"
    if set(content) <= GENERIC_ONLY:
        return "meaningless: generic words only"
    if len(set(words)) <= len(words) / 2:
        return "meaningless: repeated words"
    seed_stems = set(stem_tokens(seed))
    if c.topical_relevance < min_relevance and not (set(stem_tokens(q)) & seed_stems):
        return "unrelated to the seed topic"
    if c.topical_relevance < min_relevance * 0.6:
        return "unrelated to the seed topic (low semantic relevance)"
    if c.intent_confidence and c.intent_confidence < 0.3 and len(content) < 2:
        return "no clear search intent"
    if c.action == Action.NO_ACTION:
        return "already fully covered by an existing page (no optimisation opportunity)"
    return None


def apply_quality_control(cands: list[QueryCandidate], seed: str) -> tuple[list[QueryCandidate], list[QueryCandidate]]:
    kept, rejected = [], []
    for c in cands:
        reason = reject_reason(c, seed)
        if reason:
            c.rejected_reason = reason
            rejected.append(c)
        else:
            kept.append(c)
    log.info("Quality control: kept %d, rejected %d", len(kept), len(rejected))
    return kept, rejected


def query_tokens_present(query: str, text: str) -> float:
    """Fraction of the query's content words present in text (for 'fully covered' checks)."""
    q = {t for t in tokens(query) if t not in STOPWORDS}
    if not q:
        return 0.0
    t = set(tokens(text))
    return len(q & t) / len(q)
