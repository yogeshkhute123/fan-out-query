"""XML sitemap / sitemap-index discovery and parsing."""

from __future__ import annotations

import gzip
from collections import deque

from lxml import etree

from ..http import PoliteFetcher
from ..logging_setup import get_logger

log = get_logger("sitemap")

DEFAULT_LOCATIONS = ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml", "/wp-sitemap.xml")


def parse_sitemap_xml(content: str | bytes) -> tuple[list[str], list[str]]:
    """Return (page_urls, child_sitemaps). Tolerates malformed XML."""
    if isinstance(content, str):
        content = content.encode("utf-8", errors="ignore")
    if content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except OSError:
            return [], []
    content = content.strip()
    if not content:
        return [], []
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=False)
    try:
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError:
        return [], []
    if root is None:
        return [], []
    tag = etree.QName(root).localname.lower()
    locs = [
        (el.text or "").strip()
        for el in root.iter()
        if isinstance(el.tag, str) and etree.QName(el).localname.lower() == "loc" and el.text
    ]
    if tag == "sitemapindex":
        return [], locs
    if tag == "urlset":
        # Only <url><loc>, not image:loc etc.
        pages = []
        for url_el in root:
            if not isinstance(url_el.tag, str) or etree.QName(url_el).localname.lower() != "url":
                continue
            for child in url_el:
                if isinstance(child.tag, str) and etree.QName(child).localname.lower() == "loc" and child.text:
                    pages.append(child.text.strip())
                    break
        return pages, []
    return [], []


def discover_sitemap_urls(
    base_url: str,
    fetcher: PoliteFetcher,
    robots_sitemaps: list[str],
    max_urls: int = 5000,
    max_sitemaps: int = 50,
) -> tuple[list[str], list[str]]:
    """Walk sitemaps (robots-declared first, then common locations).

    Returns (page_urls, sitemap_urls_processed).
    """
    queue: deque[str] = deque(robots_sitemaps)
    if not queue:
        queue.extend(base_url.rstrip("/") + loc for loc in DEFAULT_LOCATIONS)
    seen: set[str] = set()
    processed: list[str] = []
    pages: list[str] = []
    page_set: set[str] = set()
    while queue and len(processed) < max_sitemaps and len(pages) < max_urls:
        sm_url = queue.popleft()
        if sm_url in seen:
            continue
        seen.add(sm_url)
        result = fetcher.fetch(sm_url)
        if not result.ok or not result.text:
            if result.status not in (404, None):
                log.info("Sitemap %s returned HTTP %s", sm_url, result.status)
            continue
        body: str | bytes = result.text
        if sm_url.endswith(".gz"):
            body = result.text.encode("latin-1", errors="ignore")
        urls, children = parse_sitemap_xml(body)
        if not urls and not children:
            log.warning("Invalid or empty sitemap: %s", sm_url)
            continue
        processed.append(sm_url)
        for child in children:
            if child not in seen:
                queue.append(child)
        for u in urls:
            if u not in page_set:
                page_set.add(u)
                pages.append(u)
                if len(pages) >= max_urls:
                    break
    log.info("Sitemaps processed: %d, URLs discovered: %d", len(processed), len(pages))
    return pages, processed
