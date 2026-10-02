"""HTML page parsing: on-page SEO elements, structured data and page type."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from ..logging_setup import get_logger
from ..models import PageData, PageType
from ..textutil import clean_text, is_question, sha1
from .urls import is_same_site, normalize_url, url_depth

log = get_logger("parser")

try:  # trafilatura is the preferred main-content extractor
    import trafilatura  # type: ignore

    _HAS_TRAFILATURA = True
except Exception:  # pragma: no cover - optional
    _HAS_TRAFILATURA = False


# ----------------------------------------------------------------- structured data
def _iter_jsonld(soup: BeautifulSoup) -> list[dict]:
    items: list[dict] = []
    for tag in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Some sites emit trailing commas / multiple objects – try a lenient fix.
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
            except json.JSONDecodeError:
                log.debug("Malformed JSON-LD skipped")
                continue
        stack: list[Any] = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                items.append(node)
                if "@graph" in node:
                    stack.append(node["@graph"])
                for key in ("mainEntity", "itemListElement", "hasPart", "offers", "about"):
                    if key in node and isinstance(node[key], (dict, list)):
                        stack.append(node[key])
    return items


def _types_of(node: dict) -> list[str]:
    t = node.get("@type")
    if isinstance(t, list):
        return [str(x) for x in t]
    return [str(t)] if t else []


def extract_structured(soup: BeautifulSoup) -> dict[str, Any]:
    nodes = _iter_jsonld(soup)
    schema_types: list[str] = []
    faq: list[str] = []
    breadcrumbs: list[str] = []
    product: dict[str, Any] = {}
    locations: list[str] = []
    for node in nodes:
        types = _types_of(node)
        schema_types.extend(types)
        if "Question" in types and node.get("name"):
            faq.append(clean_text(str(node["name"])))
        if "BreadcrumbList" in types:
            elements = node.get("itemListElement") or []
            if isinstance(elements, dict):
                elements = [elements]
            crumbs = []
            for el in sorted(
                (e for e in elements if isinstance(e, dict)), key=lambda e: int(e.get("position", 0) or 0)
            ):
                item = el.get("item")
                name = el.get("name") or (item.get("name") if isinstance(item, dict) else None)
                if name:
                    crumbs.append(clean_text(str(name)))
            if crumbs:
                breadcrumbs = crumbs
        if "Product" in types and not product:
            offers = node.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            brand = node.get("brand")
            if isinstance(brand, dict):
                brand = brand.get("name")
            product = {
                "name": clean_text(str(node.get("name") or "")),
                "brand": clean_text(str(brand or "")),
                "sku": str(node.get("sku") or ""),
                "price": str(offers.get("price") or offers.get("lowPrice") or "") if isinstance(offers, dict) else "",
                "currency": str(offers.get("priceCurrency") or "") if isinstance(offers, dict) else "",
                "availability": str(offers.get("availability") or "").rsplit("/", 1)[-1] if isinstance(offers, dict) else "",
                "rating": str((node.get("aggregateRating") or {}).get("ratingValue") or "")
                if isinstance(node.get("aggregateRating"), dict) else "",
            }
        if any(t in types for t in ("LocalBusiness", "Store", "Place", "PostalAddress")):
            addr = node.get("address") if "PostalAddress" not in types else node
            if isinstance(addr, dict):
                loc = ", ".join(
                    str(addr.get(k)) for k in ("addressLocality", "addressRegion", "addressCountry") if addr.get(k)
                )
                if loc:
                    locations.append(loc)
    # Microdata itemtypes
    for el in soup.find_all(attrs={"itemtype": True}):
        itemtype = str(el.get("itemtype", "")).rstrip("/").rsplit("/", 1)[-1]
        if itemtype:
            schema_types.append(itemtype)
    return {
        "schema_types": sorted(set(schema_types)),
        "faq": faq,
        "breadcrumbs": breadcrumbs,
        "product": product,
        "locations": locations,
    }


# --------------------------------------------------------------------- page type
_TYPE_PATTERNS: list[tuple[PageType, re.Pattern]] = [
    (PageType.PRODUCT, re.compile(r"/(products?|p|item|shop/[^/]+/[^/]+|dp)/[^/]+", re.I)),
    (PageType.COLLECTION, re.compile(r"/(collections?)(/|$)", re.I)),
    (PageType.CATEGORY, re.compile(r"/(category|categories|product-category|c|shop|department|range)(/|$)", re.I)),
    (PageType.FAQ, re.compile(r"/(faqs?|frequently-asked-questions|help-center|questions)(/|$)", re.I)),
    (PageType.GUIDE, re.compile(r"/(guides?|how-to|learn|resources|academy|tutorials?|knowledge(-base)?)(/|$)", re.I)),
    (PageType.BLOG, re.compile(r"/(blog|blogs|news|articles?|posts?|journal|magazine|stories|insights)(/|$)|/\d{4}/\d{2}/", re.I)),
    (PageType.SERVICE, re.compile(r"/(services?|solutions?|treatments?|what-we-do)(/|$)", re.I)),
    (PageType.LOCATION, re.compile(r"/(locations?|stores?|branches|near-me|cities|areas?-we-serve|store-locator)(/|$)", re.I)),
    (PageType.LANDING, re.compile(r"/(lp|landing|offers?|campaigns?|promo)(/|$)", re.I)),
]


def classify_page_type(url: str, schema_types: list[str], h1: str, product: dict, n_product_links: int) -> PageType:
    path = urlsplit(url).path
    if path in ("", "/"):
        return PageType.HOMEPAGE
    st = {s.lower() for s in schema_types}
    if "product" in st and product.get("name") and n_product_links < 6:
        return PageType.PRODUCT
    segments = [s for s in path.split("/") if s]
    for page_type, pattern in _TYPE_PATTERNS:
        if pattern.search(path):
            if page_type in (PageType.BLOG, PageType.GUIDE) and len(segments) == 1:
                return PageType.CATEGORY  # section index such as /blog/ lists posts

            # /collections/x/products/y is a product
            if page_type == PageType.COLLECTION and re.search(r"/products?/", path):
                return PageType.PRODUCT
            return page_type
    if "faqpage" in st:
        return PageType.FAQ
    if {"collectionpage", "itemlist"} & st and n_product_links >= 4:
        return PageType.CATEGORY
    if {"blogposting", "newsarticle", "article"} & st:
        return PageType.BLOG
    if {"howto"} & st:
        return PageType.GUIDE
    if {"service"} & st:
        return PageType.SERVICE
    if {"localbusiness", "store"} & st:
        return PageType.LOCATION
    if n_product_links >= 8:
        return PageType.CATEGORY
    h = h1.lower()
    if re.search(r"\b(guide|how to|tips|explained|what is)\b", h):
        return PageType.GUIDE
    return PageType.OTHER


# ------------------------------------------------------------------- main parser
def _main_content(html: str, soup: BeautifulSoup) -> str:
    text = ""
    if _HAS_TRAFILATURA:
        try:
            text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True) or ""
        except Exception as exc:  # malformed HTML can break extractors
            log.debug("trafilatura failed: %s", exc)
    if len(text) < 200:
        container = soup.find("main") or soup.find("article") or soup.body or soup
        clone = BeautifulSoup(str(container), "lxml")
        for tag in clone(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]):
            tag.decompose()
        fallback = clean_text(clone.get_text(" "))
        if len(fallback) > len(text):
            text = fallback
    return clean_text(text)


def _breadcrumbs_from_html(soup: BeautifulSoup) -> list[str]:
    nav = soup.find(attrs={"aria-label": re.compile("breadcrumb", re.I)}) or soup.find(
        class_=re.compile(r"breadcrumb", re.I)
    )
    if not nav:
        return []
    items = [clean_text(a.get_text(" ")) for a in nav.find_all(["a", "span", "li"]) if clean_text(a.get_text(" "))]
    out: list[str] = []
    for item in items:
        if item not in out and len(item) < 80 and item not in {"/", ">", "»", "›"}:
            out.append(item)
    return out[:8]


def _faq_from_html(soup: BeautifulSoup) -> list[str]:
    questions: list[str] = []
    for summary in soup.find_all("summary"):
        text = clean_text(summary.get_text(" "))
        if 8 < len(text) < 200:
            questions.append(text)
    for el in soup.find_all(class_=re.compile(r"(faq|accordion).*(question|title|header)|question", re.I)):
        text = clean_text(el.get_text(" "))
        if 8 < len(text) < 200 and is_question(text):
            questions.append(text)
    return questions


def parse_html(url: str, html: str, base_host: str, status: int | None = 200, headers: dict | None = None) -> PageData:
    headers = headers or {}
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover - lxml is robust, html.parser as last resort
        soup = BeautifulSoup(html, "html.parser")

    # canonical / robots
    canonical = None
    link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
    if link and link.get("href"):
        canonical = normalize_url(link["href"], base=url)
    robots_meta = ""
    for meta in soup.find_all("meta", attrs={"name": re.compile(r"^(robots|googlebot)$", re.I)}):
        robots_meta = (robots_meta + "," + str(meta.get("content", ""))).strip(",")
    x_robots = headers.get("x-robots-tag", "")
    directives = (robots_meta + "," + x_robots).lower()

    indexable, reason = True, "indexable"
    if status is not None and status >= 300:
        indexable, reason = False, f"HTTP {status}"
    elif "noindex" in directives or "none" in [d.strip() for d in directives.split(",")]:
        indexable, reason = False, "noindex"
    elif canonical and canonical.rstrip("/") != normalize_url(url).rstrip("/"):  # type: ignore[union-attr]
        indexable, reason = False, f"canonicalised to {canonical}"

    title = clean_text(soup.title.get_text(" ")) if soup.title else ""
    meta_desc_tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    meta_description = clean_text(meta_desc_tag.get("content", "")) if meta_desc_tag else ""
    h1 = [clean_text(h.get_text(" ")) for h in soup.find_all("h1") if clean_text(h.get_text(" "))]
    h2 = [clean_text(h.get_text(" ")) for h in soup.find_all("h2") if clean_text(h.get_text(" "))]
    h3 = [clean_text(h.get_text(" ")) for h in soup.find_all("h3") if clean_text(h.get_text(" "))]
    image_alts = [clean_text(i.get("alt", "")) for i in soup.find_all("img") if clean_text(i.get("alt", ""))]
    html_lang = (soup.html.get("lang", "") if soup.html else "") or ""

    internal: list[str] = []
    anchors: dict[str, str] = {}
    external: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = normalize_url(a["href"], base=url)
        if not href or href in seen:
            continue
        seen.add(href)
        if is_same_site(href, base_host):
            internal.append(href)
            anchor = clean_text(a.get_text(" "))[:120]
            if anchor:
                anchors[href] = anchor
        else:
            external.append(href)

    structured = extract_structured(soup)
    content = _main_content(html, soup)
    faq = list(dict.fromkeys(structured["faq"] + _faq_from_html(soup)))
    breadcrumbs = structured["breadcrumbs"] or _breadcrumbs_from_html(soup)

    product_links = [u for u in internal if re.search(r"/(products?|p|item)/[^/]+", urlsplit(u).path)]
    page_type = classify_page_type(url, structured["schema_types"], h1[0] if h1 else title, structured["product"], len(product_links))

    category_info: dict[str, Any] = {}
    if page_type in (PageType.CATEGORY, PageType.COLLECTION):
        category_info = {
            "name": h1[0] if h1 else title,
            "product_links": len(product_links),
            "breadcrumbs": breadcrumbs,
        }
    product_info = dict(structured["product"])
    if structured["locations"]:
        category_info["locations"] = structured["locations"]

    return PageData(
        url=url,
        status=status,
        canonical=canonical,
        indexable=indexable,
        index_reason=reason,
        robots_meta=robots_meta,
        title=title,
        meta_description=meta_description,
        h1=h1,
        h2=h2[:60],
        h3=h3[:100],
        main_content=content[:60_000],
        word_count=len(content.split()),
        image_alts=image_alts[:80],
        breadcrumbs=breadcrumbs,
        internal_links=internal[:800],
        internal_anchors=dict(list(anchors.items())[:800]),
        external_links=external[:300],
        schema_types=structured["schema_types"],
        faq_questions=faq[:60],
        product_info=product_info,
        category_info=category_info,
        page_type=page_type,
        depth=url_depth(url),
        content_hash=sha1(content[:20_000]),
        lang=html_lang,
    )
