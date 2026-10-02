"""Balanced final selection (MMR-style diversity with distribution caps)."""

from __future__ import annotations

import math
from collections import Counter

from ..analysis.embeddings import SemanticModel
from ..logging_setup import get_logger
from ..models import QueryCandidate

log = get_logger("selection")


def select_final(cands: list[QueryCandidate], n: int, model: SemanticModel,
                 diversity: float = 0.35) -> tuple[list[QueryCandidate], dict]:
    """Pick up to ``n`` queries balancing score with diversity across clusters/intents/types.

    Caps prevent any single cluster, intent or query type from dominating; they
    are relaxed once (adaptive to the topic) but the list is never padded with
    low-quality queries.
    """
    if not cands:
        return [], {"requested": n, "selected": 0, "pool": 0}
    emb = model.encode([c.query for c in cands])
    pool = list(range(len(cands)))
    selected: list[int] = []
    info = {"requested": n, "pool": len(cands), "cap_relaxations": 0}

    def caps(mult: float) -> dict[str, int]:
        return {
            "cluster": max(3, math.ceil(n * 0.15 * mult)),
            "intent": max(5, math.ceil(n * 0.40 * mult)),
            "type": max(5, math.ceil(n * 0.35 * mult)),
        }

    for mult in (1.0, 1.6):
        limit = caps(mult)
        cluster_count = Counter(cands[i].cluster for i in selected)
        intent_count = Counter(cands[i].intent for i in selected)
        type_count = Counter(cands[i].query_type for i in selected)
        while len(selected) < n:
            best_idx, best_val = None, -1e9
            for i in pool:
                if i in selected:
                    continue
                c = cands[i]
                if (cluster_count[c.cluster] >= limit["cluster"] or intent_count[c.intent] >= limit["intent"]
                        or type_count[c.query_type] >= limit["type"]):
                    continue
                redundancy = max((float(emb[i] @ emb[j]) for j in selected), default=0.0)
                val = c.score - diversity * 100 * max(redundancy - 0.3, 0)
                if val > best_val:
                    best_idx, best_val = i, val
            if best_idx is None:
                break
            selected.append(best_idx)
            c = cands[best_idx]
            cluster_count[c.cluster] += 1
            intent_count[c.intent] += 1
            type_count[c.query_type] += 1
        if len(selected) >= n:
            break
        info["cap_relaxations"] += 1

    final = [cands[i] for i in selected]
    final.sort(key=lambda c: -c.score)
    info["selected"] = len(final)
    if len(final) < n:
        info["shortfall_reason"] = (
            f"Only {len(final)} genuinely useful, distinct queries survived validation, de-duplication and "
            f"quality control (requested {n}). The list was not padded with low-quality variations."
        )
    log.info("Final selection: %s", info)
    return final, info
