"""Search-need de-duplication.

Removes exact duplicates, punctuation/plural/spelling variants, word
permutations and semantic near-duplicates *with the same intent*. Queries that
share words but express different needs are kept (guarded by intent and
question-frame checks).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..analysis.embeddings import SemanticModel
from ..analysis.intent import classify_intent
from ..logging_setup import get_logger
from ..models import DataLabel, QueryCandidate
from ..textutil import STOPWORDS, canonical_tokens, singularize, tokens

log = get_logger("dedup")

# Modifiers that rarely change the underlying need (country handled separately).
NEUTRAL_MODIFIERS = {
    "my", "your",
    "online", "best", "top", "good", "great", "2023", "2024", "2025", "2026", "2027", "latest", "new", "list",
    "ever", "really", "actually", "available", "right", "now", "today", "guide",
}
CONVERSATIONAL_FILLER = {"consider", "people", "make", "know", "thing", "things", "look", "important", "option", "options"}
FRAME_WORDS = {
    "what": "describe", "which": "describe", "why": "why", "how": "how", "when": "when", "where": "where", "who": "who",
    "can": "yesno", "does": "yesno", "do": "yesno", "is": "yesno", "are": "yesno", "should": "yesno", "will": "yesno",
}


def _frame(q: str) -> str:
    words = q.split()
    if not words:
        return ""
    if words[0] == "how" and len(words) > 1:
        if words[1] in {"to", "do", "can", "should"}:
            return "how-to"
        if words[1] in {"often", "much", "many", "long"}:
            return f"how-{words[1]}"
    return FRAME_WORDS.get(words[0], "describe")


def core_signature(q: str, country_terms: set[str]) -> str:
    toks = canonical_tokens(q)
    core = [t for t in toks if t not in STOPWORDS and t not in NEUTRAL_MODIFIERS and t not in CONVERSATIONAL_FILLER
            and t not in country_terms and len(t) > 1]
    return " ".join(sorted(set(core)))


def _keeper(a: QueryCandidate, b: QueryCandidate) -> tuple[QueryCandidate, QueryCandidate]:
    """Choose which candidate survives: real evidence > more sources > more natural (shorter)."""
    rank = {DataLabel.REAL: 0, DataLabel.LLM: 1, DataLabel.TEMPLATE: 2}

    def key(c: QueryCandidate):
        return (rank.get(c.label, 3), -len(c.origins), len(c.query))

    return (a, b) if key(a) <= key(b) else (b, a)


def _merge(keep: QueryCandidate, drop: QueryCandidate) -> None:
    keep.origins |= drop.origins
    for e in drop.evidence:
        if e not in keep.evidence and len(keep.evidence) < 10:
            keep.evidence.append(e)
    if len(keep.evidence) < 12:
        keep.evidence.append(f"merged duplicate: '{drop.query}'")


def deduplicate(cands: list[QueryCandidate], model: SemanticModel, country: str = "",
                semantic_threshold: float | None = None) -> tuple[list[QueryCandidate], dict[str, int]]:
    stats = {"input": len(cands), "exact_or_variant": 0, "permutation": 0, "spelling": 0, "semantic": 0}
    country_terms = set(tokens(country)) | {"india", "usa", "uk"} if country else set()
    if semantic_threshold is None:
        semantic_threshold = 0.86 if model.backend.startswith("sentence") else 0.82

    # Pass 1: plural/punctuation variants (same singularised token sequence) and permutations / neutral modifiers
    by_sig: dict[str, QueryCandidate] = {}
    survivors: list[QueryCandidate] = []
    for c in cands:
        intent = classify_intent(c.query).intent
        seq_key = " ".join(singularize(t) for t in tokens(c.query))
        sig = f"{intent.value}|{_frame(c.query)}|{core_signature(c.query, country_terms)}"
        existing = by_sig.get("seq:" + seq_key) or by_sig.get(sig)
        if existing is not None:
            keep, drop = _keeper(existing, c)
            _merge(keep, drop)
            stats["exact_or_variant" if by_sig.get("seq:" + seq_key) else "permutation"] += 1
            if keep is c:
                survivors[survivors.index(existing)] = c
                by_sig["seq:" + seq_key] = c
                by_sig[sig] = c
            continue
        by_sig["seq:" + seq_key] = c
        by_sig[sig] = c
        survivors.append(c)

    # Pass 2: spelling variants + semantic near-duplicates with identical intent and question frame
    if len(survivors) > 1:
        emb = model.encode([c.query for c in survivors])
        sim = emb @ emb.T
        intents = [classify_intent(c.query).intent for c in survivors]
        frames = [_frame(c.query) for c in survivors]
        removed: set[int] = set()
        order = sorted(range(len(survivors)), key=lambda i: survivors[i].label != DataLabel.REAL)
        for i in order:
            if i in removed:
                continue
            for j in range(len(survivors)):
                if j == i or j in removed:
                    continue
                if intents[i] != intents[j] or frames[i] != frames[j]:
                    continue
                a, b = survivors[i].query, survivors[j].query
                spelling = SequenceMatcher(None, a, b).ratio() >= 0.93 and abs(len(a) - len(b)) <= 3
                # differing content words must be neutral for a semantic merge
                diff = set(tokens(a)) ^ set(tokens(b))
                diff_core = {singularize(t) for t in diff if t not in STOPWORDS and t not in NEUTRAL_MODIFIERS and t not in country_terms}
                semantic = float(sim[i, j]) >= semantic_threshold and len(diff_core) <= 1 and not _negation_diff(a, b)
                if spelling or semantic:
                    keep, drop = _keeper(survivors[i], survivors[j])
                    _merge(keep, drop)
                    drop_idx = j if drop is survivors[j] else i
                    removed.add(drop_idx)
                    stats["spelling" if spelling else "semantic"] += 1
                    if drop_idx == i:
                        break
        survivors = [c for k, c in enumerate(survivors) if k not in removed]
    stats["output"] = len(survivors)
    stats["removed"] = stats["input"] - stats["output"]
    log.info("Deduplication: %s", stats)
    return survivors, stats


def _negation_diff(a: str, b: str) -> bool:
    neg = re.compile(r"\b(not|no|without|never|don't|doesn't|isn't|can't|free)\b")
    return bool(neg.search(a)) != bool(neg.search(b))
