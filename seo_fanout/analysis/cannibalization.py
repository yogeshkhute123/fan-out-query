"""Keyword-cannibalization detection.

Principle: *same intent + overlapping topical purpose + competing page
targeting* = potential cannibalization. Sharing a keyword is not enough, and
different keywords can still express the same need. Signals are combined into
a score, but the risk class additionally requires intent agreement.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..logging_setup import get_logger
from ..models import CannibalizationPair, PageData, PageInventory, PageType, Risk
from ..textutil import SKLEARN_STOPWORDS, canonical_text, canonical_tokens, jaccard, strip_brand_suffix
from .embeddings import SemanticModel
from .intent import FUNNEL

log = get_logger("cannibalization")

WEIGHTS = {
    "topic": 0.28,
    "title": 0.14,
    "h1": 0.10,
    "content": 0.14,
    "entity": 0.10,
    "keyword": 0.09,
    "intent": 0.15,
}

LISTING = {PageType.CATEGORY, PageType.COLLECTION}
EDITORIAL = {PageType.BLOG, PageType.GUIDE, PageType.FAQ, PageType.OTHER}


def _type_factor(a: PageType, b: PageType) -> tuple[float, str]:
    """How plausible is it that these two page types compete for one SERP?"""
    if PageType.HOMEPAGE in (a, b):
        return 0.55, "one page is the homepage (brand/navigational)"
    if a == b:
        return 1.0, "same page type"
    if {a, b} <= EDITORIAL:
        return 0.95, "both editorial pages"
    if {a, b} <= LISTING:
        return 1.0, "both listing pages"
    if (a in LISTING and b == PageType.PRODUCT) or (b in LISTING and a == PageType.PRODUCT):
        return 0.6, "listing vs product page (different scope)"
    if (a in LISTING and b in EDITORIAL) or (b in LISTING and a in EDITORIAL):
        return 0.8, "listing vs editorial page"
    return 0.85, "different page types"


def _intent_similarity(a: PageInventory, b: PageInventory) -> float:
    if a.search_intent == b.search_intent:
        return 1.0
    if FUNNEL[a.search_intent] == FUNNEL[b.search_intent]:
        return 0.6
    return 0.2


def _is_parent_child(url_a: str, url_b: str) -> bool:
    pa = urlsplit(url_a).path.rstrip("/")
    pb = urlsplit(url_b).path.rstrip("/")
    return bool(pa and pb) and (pa.startswith(pb + "/") or pb.startswith(pa + "/"))


def _strength(inv: PageInventory, page: PageData, max_words: int, max_inlinks: int) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    wc = inv.word_count / max(max_words, 1)
    score += 0.35 * wc
    il = inv.inlinks / max(max_inlinks, 1)
    score += 0.30 * il
    score += 0.10 * min(len(page.h2) / 10, 1.0)
    score += 0.10 * (1.0 if page.schema_types else 0.0)
    score += 0.15 * (1.0 / (1 + max(page.depth, 0)))
    reasons.append(f"{inv.word_count} words")
    reasons.append(f"{inv.inlinks} internal inlinks")
    reasons.append(f"{len(page.h2)} H2 sections")
    if page.schema_types:
        reasons.append("structured data: " + ", ".join(page.schema_types[:3]))
    return score, reasons


def _recommend(pair: CannibalizationPair, a_type: PageType, b_type: PageType, parent_child: bool) -> str:
    if pair.risk == Risk.HIGH:
        if a_type == PageType.PRODUCT and b_type == PageType.PRODUCT:
            return ("Consolidate: these look like product variants competing for the same query – canonicalise "
                    "variants to the main product or merge them into one page with variant options.")
        if pair.content_similarity >= 0.55:
            return ("Merge pages: consolidate the unique content into the stronger URL and 301-redirect the "
                    "weaker URL to it; update internal links to point to the surviving page.")
        return ("Retarget one page: keep the stronger URL on the shared topic and change the weaker page's "
                "search intent/angle (title, H1, sections) so each page serves a distinct need; link the "
                "weaker page to the stronger one with a descriptive anchor.")
    if pair.risk == Risk.MEDIUM:
        if parent_child:
            return ("Improve internal linking: treat the parent as the hub and the child as a spoke – make the "
                    "child clearly narrower (title/H1) and link both ways with descriptive anchors.")
        return ("Differentiate content: sharpen each page's primary topic and title so they target different "
                "angles, and add contextual internal links between them.")
    if pair.risk == Risk.LOW:
        return "Keep both pages: intents are sufficiently different; add a contextual cross-link where helpful."
    return "No action."


def detect_cannibalization(
    pages: list[PageData], inventory: list[PageInventory], model: SemanticModel
) -> tuple[list[CannibalizationPair], np.ndarray]:
    """Compare every indexable page against every other page.

    Returns (pairs with risk != NONE sorted by score, topic similarity matrix).
    """
    page_by_url = {(p.final_url or p.url): p for p in pages}
    inv = [i for i in inventory if i.url in page_by_url]
    n = len(inv)
    if n < 2:
        return [], np.zeros((n, n))

    titles = [strip_brand_suffix(i.title) for i in inv]
    h1s = [i.h1 or titles[k] for k, i in enumerate(inv)]
    topics = [i.primary_topic for i in inv]

    topic_m = model.similarity(topics, topics)
    title_m = model.similarity(titles, titles)
    h1_m = model.similarity(h1s, h1s)

    contents = [canonical_text(page_by_url[i.url].main_content[:20_000] or titles[k]) for k, i in enumerate(inv)]
    try:
        vec = TfidfVectorizer(stop_words=SKLEARN_STOPWORDS, sublinear_tf=True, max_features=40_000, ngram_range=(1, 2))
        cm = normalize(vec.fit_transform(contents))
        content_m = (cm @ cm.T).toarray()
    except ValueError:
        content_m = np.zeros((n, n))

    max_words = max((i.word_count for i in inv), default=1)
    max_inlinks = max((i.inlinks for i in inv), default=1)
    kw_sets = [{t for t in canonical_tokens(i.primary_topic + " " + titles[k]) if t not in SKLEARN_STOPWORDS}
              for k, i in enumerate(inv)]

    pairs: list[CannibalizationPair] = []
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = inv[a], inv[b]
            # cheap pre-filter: skip clearly unrelated pages
            if topic_m[a, b] < 0.2 and title_m[a, b] < 0.2 and content_m[a, b] < 0.3:
                continue
            ent = jaccard(ia.entities, ib.entities)
            kw = jaccard(kw_sets[a], kw_sets[b])
            intent_sim = _intent_similarity(ia, ib)
            base = (
                WEIGHTS["topic"] * topic_m[a, b]
                + WEIGHTS["title"] * title_m[a, b]
                + WEIGHTS["h1"] * h1_m[a, b]
                + WEIGHTS["content"] * content_m[a, b]
                + WEIGHTS["entity"] * ent
                + WEIGHTS["keyword"] * kw
                + WEIGHTS["intent"] * intent_sim
            )
            factor, type_note = _type_factor(ia.page_type, ib.page_type)
            parent_child = _is_parent_child(ia.url, ib.url)
            if parent_child:
                factor *= 0.85
            combined = float(base * factor)
            topical = float(max(topic_m[a, b], (title_m[a, b] + h1_m[a, b]) / 2))

            if combined >= 0.60 and intent_sim >= 0.99 and topical >= 0.60:
                risk = Risk.HIGH
            elif combined >= 0.47 and intent_sim >= 0.6 and topical >= 0.45:
                risk = Risk.MEDIUM
            elif combined >= 0.36 and topical >= 0.35:
                risk = Risk.LOW
            else:
                continue

            pa, pb = page_by_url[ia.url], page_by_url[ib.url]
            sa, ra = _strength(ia, pa, max_words, max_inlinks)
            sb, rb = _strength(ib, pb, max_words, max_inlinks)
            stronger, reasons = (ia.url, ra) if sa >= sb else (ib.url, rb)
            margin = abs(sa - sb)

            overlap_terms = sorted(kw_sets[a] & kw_sets[b])
            why_parts = [
                f"topic similarity {topic_m[a, b]:.2f}",
                f"title similarity {title_m[a, b]:.2f}",
                f"content similarity {content_m[a, b]:.2f}",
                f"entity overlap {ent:.2f}",
                f"intents: {ia.search_intent.value} vs {ib.search_intent.value}",
                type_note,
            ]
            if parent_child:
                why_parts.append("URLs are parent/child (hub-spoke structure)")
            if risk == Risk.LOW and intent_sim < 0.99:
                why_parts.append("topics overlap but the search intents differ")

            pair = CannibalizationPair(
                url_a=ia.url, url_b=ib.url,
                topic_similarity=round(float(topic_m[a, b]), 3),
                intent_similarity=round(intent_sim, 3),
                title_similarity=round(float(title_m[a, b]), 3),
                h1_similarity=round(float(h1_m[a, b]), 3),
                content_similarity=round(float(content_m[a, b]), 3),
                entity_overlap=round(ent, 3),
                keyword_overlap=round(kw, 3),
                combined_score=round(combined, 3),
                risk=risk,
                overlapping_topic=" ".join(overlap_terms) or ia.semantic_topic,
                overlapping_intent=ia.search_intent.value if intent_sim >= 0.99 else f"{ia.search_intent.value} / {ib.search_intent.value}",
                why="; ".join(why_parts),
                stronger_url=stronger,
                stronger_reason=("based on " + ", ".join(reasons)) + ("" if margin > 0.05 else " (signals are close – verify with Search Console data)"),
            )
            pair.recommendation = _recommend(pair, ia.page_type, ib.page_type, parent_child)
            pairs.append(pair)

    order = {Risk.HIGH: 0, Risk.MEDIUM: 1, Risk.LOW: 2}
    pairs.sort(key=lambda p: (order[p.risk], -p.combined_score))
    counts = {r: sum(1 for p in pairs if p.risk == r) for r in (Risk.HIGH, Risk.MEDIUM, Risk.LOW)}
    log.info("Cannibalization: HIGH=%d MEDIUM=%d LOW=%d (from %d pages)", counts[Risk.HIGH], counts[Risk.MEDIUM], counts[Risk.LOW], n)
    return pairs, topic_m
