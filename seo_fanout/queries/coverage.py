"""Check every candidate query against the website inventory and assign a destination.

Coverage classes: EXISTING_PAGE, PARTIALLY_COVERED, CONTENT_GAP, CANNIBALIZATION_RISK.
Each query then receives exactly one recommended action, so the output answers
"which page should target this need?" rather than "create another page".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..analysis.embeddings import SemanticModel
from ..analysis.intent import FUNNEL, classify_intent
from ..logging_setup import get_logger
from ..models import (
    Action, CannibalizationPair, Coverage, Intent, PageData, PageInventory, PageType, QueryCandidate, Risk,
)
from ..textutil import is_question, normalize_query, strip_brand_suffix
from .quality import query_tokens_present
from .url_recommender import SiteArchitecture

log = get_logger("coverage")

EDITORIAL = {PageType.BLOG, PageType.GUIDE, PageType.FAQ, PageType.OTHER, PageType.HOMEPAGE, PageType.LANDING}
NARROW_Q = re.compile(r"^(can|does|do|is|are|should|will|how (often|much|many|long)|when|which)\b")


@dataclass
class Thresholds:
    strong: float
    partial: float
    section: float
    full: float

    @classmethod
    def for_backend(cls, backend: str) -> Thresholds:
        if backend.startswith("sentence"):
            return cls(strong=0.78, partial=0.6, section=0.74, full=0.9)
        return cls(strong=0.6, partial=0.4, section=0.55, full=0.86)


def _intent_compatible(query_intent: Intent, page: PageInventory) -> bool:
    if FUNNEL[query_intent] == page.funnel_stage:
        return True
    informational = {Intent.INFORMATIONAL, Intent.RESEARCH, Intent.PROBLEM_SOLVING}
    if query_intent in informational and page.page_type in EDITORIAL:
        return True
    if query_intent in {Intent.COMMERCIAL, Intent.COMPARISON} and page.page_type in EDITORIAL | {PageType.CATEGORY, PageType.COLLECTION}:
        return True
    if query_intent in {Intent.TRANSACTIONAL, Intent.TRANSACTIONAL_INVESTIGATION} and page.page_type in {
        PageType.CATEGORY, PageType.COLLECTION, PageType.PRODUCT, PageType.SERVICE, PageType.LANDING
    }:
        return True
    return False


def _is_narrow_question(q: str) -> bool:
    return bool(NARROW_Q.search(q)) or (q.startswith(("what", "why")) and len(q.split()) <= 9)


def _content_type(action: Action, intent: Intent, q: str) -> str:
    if action == Action.FAQ_SECTION:
        return "FAQ section"
    if action == Action.CATEGORY_PAGE:
        return "Category / collection page"
    if action == Action.PRODUCT_PAGE:
        return "Product page"
    if action == Action.SERVICE_PAGE:
        return "Service / location page" if intent == Intent.LOCAL else "Service page"
    if action == Action.COMPARISON_PAGE:
        return "Comparison article"
    if action == Action.CANNIBALIZATION_REVIEW:
        return "Consolidation review"
    if q.startswith("how to") or q.startswith("how do i"):
        return "How-to guide"
    if intent == Intent.PROBLEM_SOLVING:
        return "Troubleshooting / problem-solving article"
    if intent in (Intent.COMMERCIAL, Intent.TRANSACTIONAL_INVESTIGATION):
        return "Buying guide / listicle"
    if intent == Intent.RESEARCH:
        return "Research / evidence article"
    if action in (Action.EXISTING_PAGE_EXPANSION, Action.EXISTING_PAGE_OPTIMIZATION):
        return "Section on existing page"
    return "Informational article"


def assess_coverage(
    cands: list[QueryCandidate],
    inventory: list[PageInventory],
    pages: list[PageData],
    pairs: list[CannibalizationPair],
    model: SemanticModel,
    arch: SiteArchitecture,
) -> None:
    if not cands:
        return
    th = Thresholds.for_backend(model.backend)
    queries = [c.query for c in cands]
    q_emb = model.encode(queries)

    if not inventory:
        for c in cands:
            _assign_gap(c, arch, None, 0.0, th)
        return

    primary_texts = [
        " . ".join(dict.fromkeys([i.primary_topic, normalize_query(strip_brand_suffix(i.title)), normalize_query(i.h1)]))
        for i in inventory
    ]
    primary_sim = SemanticModel.cosine_matrix(q_emb, model.encode(primary_texts))

    sections: list[tuple[int, str]] = []
    for idx, inv in enumerate(inventory):
        if inv.page_type == PageType.HOMEPAGE:
            continue  # homepage sections are teasers, not answers
        for s in inv.secondary_topics[:25] + inv.questions_answered[:25]:
            sections.append((idx, s))
    section_best = np.zeros((len(cands), len(inventory)))
    section_label: dict[tuple[int, int], str] = {}
    if sections:
        s_sim = SemanticModel.cosine_matrix(q_emb, model.encode([s for _, s in sections]))
        for col, (page_idx, text) in enumerate(sections):
            better = s_sim[:, col] > section_best[:, page_idx]
            section_best[better, page_idx] = s_sim[better, col]
            for row in np.where(better)[0]:
                section_label[(int(row), page_idx)] = text

    risky_pairs: dict[frozenset, Risk] = {
        frozenset((p.url_a, p.url_b)): p.risk for p in pairs if p.risk in (Risk.HIGH, Risk.MEDIUM)
    }
    page_by_url = {(p.final_url or p.url): p for p in pages}

    for row, c in enumerate(cands):
        ir = classify_intent(c.query)
        c.intent, c.intent_confidence = ir.intent, ir.confidence
        c.query_type, c.funnel_stage, c.user_journey = ir.query_type, ir.funnel_stage, ir.journey

        ps = primary_sim[row]
        order = np.argsort(-ps)
        c.matching_pages = [(inventory[j].url, round(float(ps[j]), 3)) for j in order[:3] if ps[j] > 0.2]
        strong = [j for j in order if ps[j] >= th.strong and _intent_compatible(ir.intent, inventory[j])]
        strong_any = [j for j in order if ps[j] >= th.strong]
        best = int(order[0])
        best_section_page = int(np.argmax(section_best[row]))
        best_section = float(section_best[row, best_section_page])

        # --- cannibalization: two+ pages strongly targeting this same need
        if len(strong) >= 2:
            a, b = inventory[strong[0]].url, inventory[strong[1]].url
            c.coverage = Coverage.CANNIBALIZATION_RISK
            c.cannibalization_risk = risky_pairs.get(frozenset((a, b)), Risk.MEDIUM)
            c.existing_page_url = a
            c.action = Action.CANNIBALIZATION_REVIEW
            c.content_gap = (f"Two pages compete for this need: {a} and {b}. Decide one primary target before "
                             f"optimising (see cannibalization report).")
        elif strong:
            j = strong[0]
            inv = inventory[j]
            c.existing_page_url = inv.url
            # The matched page may itself be in a risky pair with another page that also matches
            for other in order[1:4]:
                key = frozenset((inv.url, inventory[other].url))
                if key in risky_pairs and ps[other] >= th.partial:
                    c.cannibalization_risk = risky_pairs[key]
            if c.cannibalization_risk == Risk.HIGH:
                c.coverage = Coverage.CANNIBALIZATION_RISK
                c.action = Action.CANNIBALIZATION_REVIEW
                c.content_gap = f"Targeted by {inv.url}, which has a HIGH cannibalization risk with another page."
            else:
                c.coverage = Coverage.EXISTING_PAGE
                title_cov = query_tokens_present(c.query, inv.title + " " + inv.h1)
                page = page_by_url.get(inv.url)
                if ps[j] >= th.full and title_cov >= 0.99:
                    c.action = Action.NO_ACTION
                    c.content_gap = "Fully covered: an existing page explicitly targets this query."
                else:
                    c.action = Action.EXISTING_PAGE_OPTIMIZATION
                    missing = [t for t in normalize_query(c.query).split() if t not in normalize_query(inv.title + " " + inv.h1)]
                    c.content_gap = (f"Existing page targets this need; optimise title/H1/sections"
                                     + (f" to reflect: {' '.join(missing[:6])}" if missing else "")
                                     + (f" (thin content: {page.word_count} words)" if page and page.word_count < 400 else ""))
        elif best_section >= th.section or ps[best] >= th.partial or strong_any:
            # Covered somewhere, but not as a primary target
            j = best_section_page if best_section >= th.section else best
            # Prefer a page whose primary topic matches (with a compatible intent) over a section match
            if ps[best] >= th.partial and _intent_compatible(ir.intent, inventory[best]) and ps[best] >= best_section - 0.1:
                j = best
            if strong_any and not best_section >= th.section:
                j = strong_any[0]
            inv = inventory[j]
            c.coverage = Coverage.PARTIALLY_COVERED
            c.existing_page_url = inv.url
            if strong_any and not best_section >= th.section:
                # The topic is targeted, but by a page that serves a different intent.
                buying = ir.intent in (Intent.TRANSACTIONAL, Intent.TRANSACTIONAL_INVESTIGATION) and not is_question(c.query)
                if buying and arch.is_ecommerce:
                    c.action = Action.CATEGORY_PAGE
                elif buying or ir.intent == Intent.COMMERCIAL:
                    c.action = Action.NEW_PAGE
                else:
                    c.action = Action.EXISTING_PAGE_EXPANSION
                c.content_gap = (f"{inv.url} covers the topic but serves a different intent "
                                 f"({inv.search_intent.value}); the {ir.intent.value} need is not served directly.")
            elif _is_narrow_question(c.query):
                c.action = Action.FAQ_SECTION
                c.content_gap = f"Partially answered on {inv.url}" + (
                    f" (section: '{section_label.get((row, j), '')}')" if (row, j) in section_label else "") + "; add a direct FAQ answer."
            else:
                c.action = Action.EXISTING_PAGE_EXPANSION
                sec = section_label.get((row, j))
                c.content_gap = (f"Mentioned on {inv.url}" + (f" in section '{sec}'" if sec else "")
                                 + " but not covered in depth – expand that page; evaluate a dedicated page only if demand is large and the intent is distinct.")
        else:
            _assign_gap(c, arch, inventory[best] if ps[best] >= th.partial * 0.7 else None, float(ps[best]), th)

        c.content_type = _content_type(c.action, ir.intent, normalize_query(c.query))  # type: ignore[arg-type]
        c.recommended_url, _ = arch.recommend(c.action, c.query, c.existing_page_url)  # type: ignore[arg-type]

    counts: dict[str, int] = {}
    for c in cands:
        counts[c.coverage.value] = counts.get(c.coverage.value, 0) + 1  # type: ignore[union-attr]
    log.info("Coverage assessment: %s", counts)


def _assign_gap(c: QueryCandidate, arch: SiteArchitecture, related: PageInventory | None, sim: float, th: Thresholds) -> None:
    ir = classify_intent(c.query)
    c.intent, c.intent_confidence = ir.intent, ir.confidence
    c.query_type, c.funnel_stage, c.user_journey = ir.query_type, ir.funnel_stage, ir.journey
    c.coverage = Coverage.CONTENT_GAP
    q = normalize_query(c.query)
    if ir.intent == Intent.COMPARISON:
        c.action = Action.COMPARISON_PAGE
    elif ir.intent in (Intent.TRANSACTIONAL, Intent.TRANSACTIONAL_INVESTIGATION) and arch.is_ecommerce and not is_question(q):
        c.action = Action.PRODUCT_PAGE if (ir.intent == Intent.TRANSACTIONAL and len(q.split()) <= 4
                                           and not re.search(r"\b(best|top|brands?|products?|online)\b", q)) else Action.CATEGORY_PAGE
    elif ir.intent in (Intent.TRANSACTIONAL, Intent.LOCAL, Intent.COMMERCIAL) and arch.is_service_business and not arch.is_ecommerce:
        c.action = Action.SERVICE_PAGE
    elif related is not None and _is_narrow_question(q):
        c.action = Action.FAQ_SECTION
        c.existing_page_url = None
        c.recommended_url = related.url + "#faq"
    else:
        c.action = Action.NEW_PAGE
    c.content_gap = "No existing page addresses this search need" + (
        f"; closest page is {related.url} (similarity {sim:.2f})" if related is not None else "") + "."
    if c.action == Action.FAQ_SECTION and related is not None:
        c.content_gap += f" Answer it as an FAQ on {related.url}."
        c.existing_page_url = related.url
