"""Topic clustering of fan-out queries and the website topic map."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..logging_setup import get_logger
from ..models import Action, CannibalizationPair, Coverage, Intent, PageInventory, QueryCandidate, Risk
from ..textutil import STOPWORDS, QUESTION_WORDS, normalize_query, singularize, tokens
from .embeddings import SemanticModel

log = get_logger("topics")

LABEL_STOP = STOPWORDS | set(QUESTION_WORDS) | {
    "best", "top", "good", "vs", "versus", "need", "know", "choose", "right", "people", "make", "common", "type",
    "types", "way", "ways", "thing", "things", "work", "worth", "much", "many", "often", "long", "really",
}


def _residual(query: str, seed_tokens: set[str]) -> str:
    """Query without the seed words – clustering on what *differs* from the seed."""
    toks = [t for t in tokens(query) if singularize(t) not in seed_tokens and t not in STOPWORDS]
    return " ".join(toks) or "core"


def _label(queries: list[str], seed_tokens: set[str], fallback: str) -> str:
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    for q in queries:
        toks = [t for t in tokens(q) if singularize(t) not in seed_tokens and t not in LABEL_STOP and len(t) > 2]
        unigrams.update(set(toks))
        bigrams.update({f"{a} {b}" for a, b in zip(toks, toks[1:])})
    n = len(queries)
    if bigrams:
        bg, c = bigrams.most_common(1)[0]
        if c >= max(2, n * 0.4):
            return bg
    if unigrams:
        word, c = unigrams.most_common(1)[0]
        if c >= max(1, n * 0.3):
            second = [w for w, k in unigrams.most_common(3)[1:] if k >= max(2, n * 0.3)]
            return word + (f" & {second[0]}" if second else "")
    return fallback


def cluster_queries(cands: list[QueryCandidate], seed: str, parent_topic: str, model: SemanticModel) -> None:
    """Assign parent_topic, topic_cluster and sub_cluster to every candidate."""
    if not cands:
        return
    seed_tokens = {singularize(t) for t in tokens(seed) if t not in STOPWORDS}
    residuals = [_residual(c.query, seed_tokens) for c in cands]
    emb = model.encode(residuals)
    if len(cands) == 1:
        labels = np.array([0])
    else:
        threshold = 0.7 if model.backend.startswith("sentence") else 0.8
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
        ).fit_predict(emb)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[int(lab)].append(idx)

    # Singletons rarely form a meaningful topic on their own: fold them into need-based
    # clusters (problems, comparisons, buying...) so the map stays readable.
    named: dict[str, list[int]] = defaultdict(list)
    for members in groups.values():
        if len(members) >= 2:
            label = _label([cands[i].query for i in members], seed_tokens, "core / general")
            named[label] += members
        else:
            named[_need_label(cands[members[0]])] += members

    for label, members in named.items():
        cluster_name = f"{parent_topic} – {label}"
        for i in members:
            c = cands[i]
            c.parent_topic = parent_topic
            c.cluster = cluster_name
            c.sub_cluster = f"{label} · {c.query_type or 'general'}"
    log.info("Clustering: %d candidates -> %d clusters", len(cands), len(named))


NEED_LABELS = {
    Intent.PROBLEM_SOLVING: "problems & troubleshooting",
    Intent.COMPARISON: "comparisons & alternatives",
    Intent.COMMERCIAL: "buying guides & recommendations",
    Intent.TRANSACTIONAL_INVESTIGATION: "pricing & value",
    Intent.TRANSACTIONAL: "purchase",
    Intent.LOCAL: "local",
    Intent.NAVIGATIONAL: "brand & navigation",
    Intent.RESEARCH: "ingredients, science & research",
}


def _need_label(c: QueryCandidate) -> str:
    if c.intent in NEED_LABELS:
        return NEED_LABELS[c.intent]  # type: ignore[index]
    q = c.query
    if q.startswith(("how to", "how do", "how often", "how much", "how can")):
        return "usage & how-to"
    if q.startswith(("is ", "are ", "can ", "does ", "do ", "should ")):
        return "suitability & safety questions"
    if q.startswith(("what", "why", "who", "when", "which")):
        return "basics & understanding"
    return "core / general"


@dataclass
class TopicMapRow:
    parent_topic: str
    cluster: str
    queries: int
    existing_page: str
    covered_topics: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    overlapping_pages: list[str] = field(default_factory=list)
    content_gap: str = ""
    potential_new_page: str = ""
    intents: str = ""


def build_topic_map(cands: list[QueryCandidate], pairs: list[CannibalizationPair]) -> list[TopicMapRow]:
    by_cluster: dict[str, list[QueryCandidate]] = defaultdict(list)
    for c in cands:
        by_cluster[c.cluster].append(c)
    risky = {frozenset((p.url_a, p.url_b)) for p in pairs if p.risk in (Risk.HIGH, Risk.MEDIUM)}
    rows = []
    for cluster, members in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        covered = [c for c in members if c.coverage in (Coverage.EXISTING_PAGE, Coverage.PARTIALLY_COVERED)]
        missing = [c for c in members if c.coverage == Coverage.CONTENT_GAP]
        pages = Counter(c.existing_page_url.split("#")[0] for c in covered if c.existing_page_url)
        existing = pages.most_common(1)[0][0] if pages else ""
        overlapping = set()
        page_list = list(pages)
        for i, a in enumerate(page_list):
            for b in page_list[i + 1:]:
                if frozenset((a, b)) in risky:
                    overlapping.update((a, b))
        for c in members:
            if c.coverage == Coverage.CANNIBALIZATION_RISK and c.existing_page_url:
                overlapping.add(c.existing_page_url)
        new_page = ""
        new_page_cands = [c for c in missing if c.action in (Action.NEW_PAGE, Action.COMPARISON_PAGE, Action.CATEGORY_PAGE,
                                                               Action.SERVICE_PAGE, Action.PRODUCT_PAGE)]
        if new_page_cands:
            top = max(new_page_cands, key=lambda c: c.score)
            new_page = f"{top.recommended_url} (for: {top.query})"
        if missing and existing:
            gap = f"{len(missing)} need(s) not served; consider expanding {existing} or a dedicated page if intents differ."
        elif missing:
            gap = f"{len(missing)} need(s) not served by any page."
        else:
            gap = "No significant gap."
        rows.append(TopicMapRow(
            parent_topic=members[0].parent_topic, cluster=cluster, queries=len(members), existing_page=existing,
            covered_topics=[c.query for c in covered][:12], missing_topics=[c.query for c in missing][:12],
            overlapping_pages=sorted(overlapping), content_gap=gap, potential_new_page=new_page,
            intents=", ".join(f"{k}:{v}" for k, v in Counter(c.intent.value for c in members if c.intent).most_common()),
        ))
    return rows


def parent_topic_for(seed: str) -> str:
    s = normalize_query(seed)
    m = re.match(r"^(.+?)\s+(for|in|with|on|near|without|vs)\s+", s)
    return m.group(1) if m else s
