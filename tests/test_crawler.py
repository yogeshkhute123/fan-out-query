from seo_fanout.crawler.crawler import Crawler, WebsiteUnavailableError
from seo_fanout.crawler.parser import parse_html
from seo_fanout.crawler.sitemap import parse_sitemap_xml
from seo_fanout.crawler.urls import (
    InvalidDomainError, dedup_key, is_crawlable_path, is_pagination, normalize_url, validate_domain,
)
from seo_fanout.models import PageType

import pytest


def test_validate_domain():
    assert validate_domain("example.com") == "https://example.com"
    assert validate_domain("HTTP://Example.com/path?x=1") == "http://example.com"
    assert validate_domain("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    for bad in ("", "not a domain", "http://", "exa mple.com", "-bad.com"):
        with pytest.raises(InvalidDomainError):
            validate_domain(bad)


def test_normalize_url_strips_tracking_and_fragments():
    url = normalize_url("HTTPS://Example.com:443//a//b/index.html?utm_source=x&b=2&a=1#top")
    assert url == "https://example.com/a/b/?a=1&b=2"
    assert normalize_url("mailto:a@b.com") is None
    assert normalize_url("/x", base="https://example.com/y/") == "https://example.com/x"
    assert dedup_key("https://www.example.com/a/") == dedup_key("https://example.com/a")


def test_crawlable_and_pagination():
    assert not is_crawlable_path("https://e.com/image.jpg")
    assert not is_crawlable_path("https://e.com/cart")
    assert is_crawlable_path("https://e.com/blog/post")
    assert is_pagination("https://e.com/blog/page/2/")
    assert is_pagination("https://e.com/c?page=3")


def test_sitemap_parsing_tolerates_bad_xml():
    urls, children = parse_sitemap_xml(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://e.com/a</loc></url>'
        '<url><loc>https://e.com/b</loc></url></urlset>'
    )
    assert urls == ["https://e.com/a", "https://e.com/b"] and children == []
    urls, children = parse_sitemap_xml('<sitemapindex><sitemap><loc>https://e.com/s1.xml</loc></sitemap></sitemapindex>')
    assert children == ["https://e.com/s1.xml"]
    assert parse_sitemap_xml("<<<not xml") == ([], [])
    assert parse_sitemap_xml("") == ([], [])


def test_parser_extracts_seo_elements():
    html = """<html><head><title>Gel Sunscreen | Brand</title><meta name="description" content="Desc">
    <link rel="canonical" href="https://e.com/products/gel"><meta name="robots" content="index,follow">
    <script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"Product","name":"Gel",
    "offers":{"price":"10","priceCurrency":"USD"}},{"@type":"FAQPage","mainEntity":[{"@type":"Question","name":"Is it oily?"}]}]}</script>
    </head><body><h1>Gel Sunscreen</h1><h2>Why gel?</h2><h3>Details</h3><img src=a.png alt="tube">
    <a href="/collections/sun">Sun</a><a href="https://other.com/x">x</a><p>Some text here.</p></body></html>"""
    page = parse_html("https://e.com/products/gel", html, "e.com")
    assert page.title == "Gel Sunscreen | Brand"
    assert page.h1 == ["Gel Sunscreen"] and page.h2 == ["Why gel?"] and page.h3 == ["Details"]
    assert page.canonical == "https://e.com/products/gel" and page.indexable
    assert "Product" in page.schema_types and "FAQPage" in page.schema_types
    assert page.product_info["price"] == "10"
    assert "Is it oily?" in page.faq_questions
    assert page.page_type == PageType.PRODUCT
    assert page.image_alts == ["tube"]
    assert page.internal_links == ["https://e.com/collections/sun"]
    assert page.external_links == ["https://other.com/x"]


def test_parser_noindex_and_canonicalised():
    noindex = parse_html("https://e.com/a", "<html><head><meta name='robots' content='noindex'></head><body>x</body></html>", "e.com")
    assert not noindex.indexable and noindex.index_reason == "noindex"
    canon = parse_html("https://e.com/a-copy", "<html><head><link rel='canonical' href='https://e.com/a'></head></html>", "e.com")
    assert not canon.indexable and "canonicalised" in canon.index_reason


def test_parser_handles_malformed_html():
    page = parse_html("https://e.com/x", "<html><body><h1>Broken<div><p>unclosed <b>tags", "e.com")
    assert page.h1 and page.status == 200


def test_crawl_fixture_site(fixture_site, settings, db):
    website_id = db.get_or_create_website("127.0.0.1", fixture_site)
    crawler = Crawler(fixture_site, settings.crawl, db, website_id)
    pages = crawler.crawl()
    crawler.close()
    urls = {p.final_url or p.url for p in pages}
    assert any(u.endswith("/blog/best-sunscreen-for-oily-skin/") for u in urls)
    assert any("/products/gel-sunscreen-spf-50" in u for u in urls)
    # robots.txt disallow respected
    assert not any("/private/" in u for u in urls)
    assert crawler.stats.skipped_robots >= 1
    # noindex + canonicalised duplicate recorded but not indexable
    by_url = {p.final_url or p.url: p for p in pages}
    old = next(p for u, p in by_url.items() if "old-sunscreen-post" in u)
    assert not old.indexable
    copy = next(p for u, p in by_url.items() if "oily-skin-copy" in u)
    assert not copy.indexable
    # sitemap index -> child sitemaps
    assert len(crawler.stats.sitemaps) == 3
    # stored in SQLite
    assert len(db.load_pages(website_id)) == len(pages)


def test_crawl_respects_max_pages(fixture_site, settings, db):
    settings.crawl.max_pages = 3
    website_id = db.get_or_create_website("127.0.0.1-max", fixture_site)
    crawler = Crawler(fixture_site, settings.crawl, db, website_id)
    assert len(crawler.crawl()) == 3
    crawler.close()


def test_unavailable_website(settings, db):
    settings.crawl.max_retries = 0
    settings.crawl.timeout = 1
    website_id = db.get_or_create_website("closed", "http://127.0.0.1:9")
    crawler = Crawler("http://127.0.0.1:9", settings.crawl, db, website_id)
    with pytest.raises(WebsiteUnavailableError):
        crawler.crawl()
    crawler.close()
