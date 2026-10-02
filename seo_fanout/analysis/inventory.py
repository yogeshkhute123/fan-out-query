"""Website content inventory: what each page is about and which need it serves."""

from __future__ import annotations

import re
from collections import Counter
from urllib.parse import urlsplit

from ..crawler.urls import is_pagination
from ..logging_setup import get_logger
from ..models import Intent, PageData, PageInventory, PageType
from ..textutil import clean_text, is_question, key_phrases, normalize_query, strip_brand_suffix
from .intent import FUNNEL, classify_intent

log = get_logger("inventory")

# A deliberately small gazetteer used only to *flag* location-targeted pages.
LOCATIONS = {
    "india", "usa", "united states", "uk", "united kingdom", "canada", "australia", "germany", "france",
    "singapore", "uae", "dubai", "new york", "london", "toronto", "sydney", "melbourne", "los angeles",
    "chicago", "san francisco", "mumbai", "delhi", "new delhi", "bangalore", "bengaluru", "hyderabad",
    "chennai", "kolkata", "pune", "ahmedabad", "jaipur", "lucknow", "gurgaon", "gurugram", "noida",
    "kochi", "chandigarh", "indore", "texas", "california", "florida", "ontario", "maharashtra",
    "karnataka", "tamil nadu", "kerala", "gujarat", "punjab", "berlin", "paris", "manchester",
}

CONTENT_TYPE = {
    PageType.HOMEPAGE: "Homepage",
    PageType.CATEGORY: "Category listing",
    PageType.COLLECTION: "Collection listing",
    PageType.PRODUCT: "Product detail page",
    PageType.BLOG: "Blog article",
    PageType.GUIDE: "Guide",
    PageType.LANDING: "Landing page",
    PageType.SERVICE: "Service page",
    PageType.LOCATION: "Location page",
    PageType.FAQ: "FAQ page",
    PageType.OTHER: "Informational page",
}


def page_topic_text(page: PageData) -> str:
    """Compact representation of a page's *targeting* (not its full body)."""
    title = strip_brand_suffix(page.title)
    h1 = page.h1[0] if page.h1 else ""
    parts = [title, h1, h1, page.meta_description, " ".join(page.h2[:12]), " ".join(page.breadcrumbs[-2:])]
    return clean_text(" . ".join(p for p in parts if p))


def _primary_topic(page: PageData, site_name: str) -> str:
    h1 = page.h1[0] if page.h1 else ""
    title = strip_brand_suffix(page.title)
    candidate = h1 if 3 <= len(h1) <= 120 else title
    if page.page_type == PageType.HOMEPAGE:
        candidate = title or h1 or site_name
    candidate = re.sub(r"\s*\(\d{4}\)|\b(20\d{2})\b", "", candidate)  # drop years
    return normalize_query(candidate)


def _content_type(page: PageData, primary: str) -> str:
    base = CONTENT_TYPE[page.page_type]
    if page.page_type in (PageType.BLOG, PageType.GUIDE, PageType.OTHER):
        if re.search(r"\bvs\b|versus|compared", primary):
            return "Comparison article"
        if re.search(r"\b(best|top \d+|top)\b", primary):
            return "Listicle / buying guide"
        if primary.startswith("how to") or "guide" in primary:
            return "How-to guide"
        if is_question(primary):
            return "Q&A article"
    return base


def _intent_for(page: PageData, primary: str) -> Intent:
    pt = page.page_type
    if pt == PageType.PRODUCT:
        return Intent.TRANSACTIONAL
    if pt in (PageType.CATEGORY, PageType.COLLECTION):
        return Intent.TRANSACTIONAL_INVESTIGATION
    if pt == PageType.LOCATION:
        return Intent.LOCAL
    if pt == PageType.HOMEPAGE:
        return Intent.NAVIGATIONAL
    result = classify_intent(primary)
    if pt == PageType.SERVICE and result.intent == Intent.INFORMATIONAL:
        return Intent.COMMERCIAL
    return result.intent


def build_inventory(pages: list[PageData], site_name: str = "") -> list[PageInventory]:
    """Create one inventory row per indexable HTML page."""
    inlinks: Counter[str] = Counter()
    for p in pages:
        for link in set(p.internal_links):
            inlinks[link.rstrip("/")] += 1

    inventory: list[PageInventory] = []
    for page in pages:
        url = page.final_url or page.url
        if not page.indexable or is_pagination(url):
            continue
        primary = _primary_topic(page, site_name)
        topic_text = page_topic_text(page)
        heading_text = " . ".join([strip_brand_suffix(page.title)] + page.h1 + page.h2 + page.h3)
        entities = key_phrases(heading_text + " . " + page.main_content[:15_000], top_k=15)
        secondary = [normalize_query(h) for h in page.h2[:15] if 3 < len(h) < 120]

        questions = list(dict.fromkeys(
            [q for q in page.faq_questions] + [h for h in page.h2 + page.h3 if is_question(h)]
        ))[:40]

        products: list[str] = []
        if page.product_info.get("name"):
            products.append(page.product_info["name"])
        if page.page_type in (PageType.CATEGORY, PageType.COLLECTION):
            for link, anchor in page.internal_anchors.items():
                if re.search(r"/(products?|p|item)/", urlsplit(link).path) and 3 < len(anchor) < 80:
                    products.append(anchor)
        services = [primary] if page.page_type == PageType.SERVICE else []
        lower = normalize_query(topic_text)
        locations = sorted({loc for loc in LOCATIONS if re.search(rf"\b{re.escape(loc)}\b", lower)})
        locations += page.category_info.get("locations", [])

        intent = _intent_for(page, primary)
        likely = [primary]
        title_kw = normalize_query(strip_brand_suffix(page.title))
        if title_kw and title_kw != primary:
            likely.append(title_kw)
        primary_tokens = set(primary.split())
        likely += [e for e in entities if set(e.split()) & primary_tokens][:6]

        inventory.append(
            PageInventory(
                url=url,
                page_type=page.page_type,
                primary_topic=primary,
                secondary_topics=secondary,
                entities=entities,
                products=list(dict.fromkeys(products))[:25],
                services=services,
                locations=list(dict.fromkeys(locations)),
                questions_answered=questions,
                search_intent=intent,
                likely_keywords=list(dict.fromkeys(k for k in likely if k))[:10],
                semantic_topic=entities[0] if entities else primary,
                content_type=_content_type(page, primary),
                funnel_stage=FUNNEL[intent],
                word_count=page.word_count,
                inlinks=inlinks.get(url.rstrip("/"), 0),
                title=page.title,
                h1=page.h1[0] if page.h1 else "",
            )
        )
    log.info("Content inventory built for %d indexable pages", len(inventory))
    return inventory
