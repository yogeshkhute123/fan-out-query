"""A small, realistic skincare store used for tests and local end-to-end runs.

It deliberately contains:
  * two blog posts targeting the same need (HIGH cannibalization),
  * a collection + guide on related but different intents,
  * a noindex page, a canonicalised duplicate and a robots.txt-disallowed path,
  * a sitemap index with two child sitemaps, JSON-LD (Product, FAQPage, BreadcrumbList).
Also provides a tiny "competitor" site for competitor-analysis tests.
"""

from __future__ import annotations

import functools
import http.server
import json
import socketserver
import threading
from pathlib import Path

LOREM = (
    "Choosing skincare depends on your skin type, climate and routine. Dermatologists recommend reading labels, "
    "patch testing new products and applying enough product for real protection. "
)


def _page(title: str, h1: str, body: str, *, canonical: str | None = None, robots: str | None = None,
          jsonld: list[dict] | None = None, nav: str = "", desc: str = "") -> str:
    head = [f"<title>{title}</title>", f'<meta name="description" content="{desc or h1}">']
    if canonical:
        head.append(f'<link rel="canonical" href="{canonical}">')
    if robots:
        head.append(f'<meta name="robots" content="{robots}">')
    for block in jsonld or []:
        head.append(f'<script type="application/ld+json">{json.dumps(block)}</script>')
    menu = (
        '<nav><a href="/">Home</a> <a href="/collections/sunscreen">Sunscreen</a> '
        '<a href="/blog/">Blog</a> <a href="/pages/faq">FAQ</a> <a href="/cart">Cart</a> '
        '<a href="/blog/sunscreen-guide">Sunscreen guide</a></nav>'
    )
    return (f"<!doctype html><html lang='en'><head>{''.join(head)}</head><body>{menu}<main><h1>{h1}</h1>"
            f"{body}{nav}</main><footer><a href='/pages/about'>About</a></footer></body></html>")


def _p(n: int, extra: str = "") -> str:
    return "".join(f"<p>{extra} {LOREM}</p>" for _ in range(n))


def _faq_ld(questions: list[str]) -> dict:
    return {"@context": "https://schema.org", "@type": "FAQPage",
            "mainEntity": [{"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": "Answer."}}
                           for q in questions]}


def build_site(root: Path, base: str) -> Path:
    files: dict[str, str] = {}
    files["robots.txt"] = f"User-agent: *\nDisallow: /private/\nDisallow: /cart\n\nSitemap: {base}/sitemap_index.xml\n"
    files["sitemap_index.xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<sitemap><loc>{base}/sitemap-pages.xml</loc></sitemap><sitemap><loc>{base}/sitemap-products.xml</loc></sitemap>"
        "</sitemapindex>"
    )
    page_urls = ["/", "/blog/", "/blog/best-sunscreen-for-oily-skin", "/blog/top-sunscreens-for-oily-skin",
                 "/blog/sunscreen-guide", "/blog/acne-skin-care-guide", "/blog/how-to-apply-sunscreen",
                 "/blog/vitamin-c-serum-benefits", "/pages/faq", "/pages/about", "/collections/sunscreen",
                 "/blog/old-sunscreen-post", "/private/internal-notes", "/blog/best-sunscreen-for-oily-skin-copy"]
    product_urls = ["/products/gel-sunscreen-spf-50", "/products/matte-sunscreen-spf-30", "/products/vitamin-c-serum"]
    files["sitemap-pages.xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(f"<url><loc>{base}{u}</loc></url>" for u in page_urls) + "</urlset>"
    )
    files["sitemap-products.xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(f"<url><loc>{base}{u}</loc></url>" for u in product_urls) + "</urlset>"
    )

    files["index.html"] = _page(
        "GlowSkin | Skincare for Indian Weather", "Skincare made for Indian weather",
        "<h2>Shop sunscreen</h2><p>Lightweight sunscreens and serums.</p>" + _p(2)
        + '<a href="/products/gel-sunscreen-spf-50">Gel Sunscreen SPF 50</a>'
        + '<a href="/blog/best-sunscreen-for-oily-skin">Best sunscreen for oily skin</a>',
    )
    files["blog/index.html"] = _page(
        "Blog | GlowSkin", "GlowSkin Blog",
        "".join(f'<a href="{u}">{u}</a> ' for u in page_urls if u.startswith("/blog/") and u != "/blog/")
        + '<a href="/blog/page/2">Older posts</a>',
    )
    files["blog/page/2/index.html"] = _page("Blog page 2 | GlowSkin", "GlowSkin Blog – page 2",
                                            '<a href="/blog/old-sunscreen-post">Old post</a>')
    best_ld = [_faq_ld(["Which sunscreen is best for oily skin?", "Can oily skin use sunscreen every day?"]),
               {"@context": "https://schema.org", "@type": "BlogPosting", "headline": "Best Sunscreen for Oily Skin"}]
    files["blog/best-sunscreen-for-oily-skin/index.html"] = _page(
        "Best Sunscreen for Oily Skin in India (2026) | GlowSkin", "Best Sunscreen for Oily Skin",
        "<h2>Why oily skin needs a different sunscreen</h2>" + _p(4, "Oily skin sunscreen gel matte finish SPF.")
        + "<h2>Gel vs cream sunscreen for oily skin</h2>" + _p(3, "Gel sunscreen absorbs fast on oily skin.")
        + "<h2>Our top picks for oily skin</h2>" + _p(3, "Matte sunscreen non-comedogenic oil-free.")
        + "<h3>Which sunscreen is best for oily skin?</h3>" + _p(1)
        + "<h3>Can oily skin use sunscreen every day?</h3>" + _p(1)
        + '<a href="/products/gel-sunscreen-spf-50">Gel sunscreen</a>',
        jsonld=best_ld, desc="The best sunscreens for oily, acne-prone skin – gel and matte picks.",
    )
    files["blog/top-sunscreens-for-oily-skin/index.html"] = _page(
        "Top Sunscreens for Oily Skin – Expert Picks | GlowSkin", "Top Sunscreens for Oily Skin",
        "<h2>Best sunscreens for oily skin compared</h2>" + _p(3, "Oily skin sunscreen gel matte finish SPF.")
        + "<h2>How to pick a sunscreen for oily skin</h2>" + _p(2, "Matte sunscreen non-comedogenic oil-free.")
        + "<h2>Gel sunscreen picks</h2>" + _p(2, "Gel sunscreen absorbs fast on oily skin."),
        jsonld=[{"@context": "https://schema.org", "@type": "BlogPosting", "headline": "Top Sunscreens"}],
    )
    files["blog/best-sunscreen-for-oily-skin-copy/index.html"] = _page(
        "Best Sunscreen for Oily Skin | GlowSkin", "Best Sunscreen for Oily Skin", _p(3),
        canonical=f"{base}/blog/best-sunscreen-for-oily-skin",
    )
    files["blog/sunscreen-guide/index.html"] = _page(
        "The Complete Sunscreen Guide | GlowSkin", "The Complete Sunscreen Guide",
        "<h2>What is SPF?</h2>" + _p(2, "SPF measures UVB protection.")
        + "<h2>Chemical vs mineral sunscreen</h2>" + _p(2, "Mineral sunscreen uses zinc oxide.")
        + "<h2>How much sunscreen should you apply?</h2>" + _p(2) + "<h2>How often should you reapply sunscreen?</h2>"
        + _p(2) + "<h2>Sunscreen ingredients to know</h2>" + _p(2, "Niacinamide, zinc oxide, titanium dioxide.")
        + '<a href="/blog/how-to-apply-sunscreen">How to apply sunscreen</a>',
        jsonld=[{"@context": "https://schema.org", "@type": "Article", "headline": "Sunscreen guide"}],
    )
    files["blog/how-to-apply-sunscreen/index.html"] = _page(
        "How to Apply Sunscreen Correctly | GlowSkin", "How to Apply Sunscreen Correctly",
        "<h2>Step-by-step application</h2>" + _p(3) + "<h2>Applying sunscreen over makeup</h2>" + _p(2)
        + "<h2>Common sunscreen application mistakes</h2>" + _p(2),
    )
    files["blog/acne-skin-care-guide/index.html"] = _page(
        "Acne Skin Care Guide | GlowSkin", "Acne Skin Care Guide",
        "<h2>What causes acne?</h2>" + _p(3) + "<h2>Choosing non-comedogenic products</h2>" + _p(2)
        + "<h2>Sunscreen for acne-prone skin</h2>" + _p(2, "Oil-free sunscreen helps acne-prone oily skin."),
    )
    files["blog/vitamin-c-serum-benefits/index.html"] = _page(
        "Vitamin C Serum Benefits | GlowSkin", "Vitamin C Serum Benefits",
        "<h2>Brightening</h2>" + _p(3) + "<h2>How to use vitamin C with sunscreen</h2>" + _p(2),
    )
    files["blog/old-sunscreen-post/index.html"] = _page(
        "Old Sunscreen Post | GlowSkin", "Old Sunscreen Post", _p(2), robots="noindex, follow",
    )
    files["pages/faq/index.html"] = _page(
        "FAQ | GlowSkin", "Frequently Asked Questions",
        "<h2>Shipping</h2>" + _p(1) + "<h3>Do you ship across India?</h3>" + _p(1)
        + "<h3>Is gel sunscreen good for oily skin?</h3>" + _p(1)
        + "<details><summary>Can I return a product?</summary>Yes.</details>",
        jsonld=[_faq_ld(["Do you ship across India?", "Is gel sunscreen good for oily skin?"])],
    )
    files["pages/about/index.html"] = _page("About GlowSkin", "About GlowSkin", _p(3, "We are a skincare brand from Mumbai."))
    files["collections/sunscreen/index.html"] = _page(
        "Buy Sunscreen Online – SPF 30 & SPF 50 | GlowSkin", "Sunscreen",
        "<h2>Shop sunscreens</h2>" + "".join(f'<a href="{u}">{u.rsplit("/", 1)[-1].replace("-", " ").title()}</a> '
                                             for u in product_urls[:2]) + _p(1),
        jsonld=[{"@context": "https://schema.org", "@type": "CollectionPage", "name": "Sunscreen"},
                {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
                    {"@type": "ListItem", "position": 1, "name": "Home"},
                    {"@type": "ListItem", "position": 2, "name": "Sunscreen"}]}],
    )
    for u, name, price in (("/products/gel-sunscreen-spf-50", "Gel Sunscreen SPF 50", "499"),
                           ("/products/matte-sunscreen-spf-30", "Matte Sunscreen SPF 30", "399"),
                           ("/products/vitamin-c-serum", "Vitamin C Serum", "699")):
        files[u.lstrip("/") + "/index.html"] = _page(
            f"{name} | GlowSkin", name,
            "<h2>Product details</h2>" + _p(2, f"{name} for oily skin, lightweight.") + "<h2>How to use</h2>" + _p(1),
            jsonld=[{"@context": "https://schema.org", "@type": "Product", "name": name, "brand": {"@type": "Brand", "name": "GlowSkin"},
                     "offers": {"@type": "Offer", "price": price, "priceCurrency": "INR", "availability": "https://schema.org/InStock"}}],
        )
    files["private/internal-notes/index.html"] = _page("Private", "Private notes", _p(1))

    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def build_competitor_site(root: Path) -> Path:
    files = {
        "robots.txt": "User-agent: *\nAllow: /\n",
        "oily-skin-sunscreen/index.html": _page(
            "10 Best Sunscreens for Oily Skin | CompetitorMag", "10 Best Sunscreens for Oily Skin",
            "<h2>Why does sunscreen make oily skin greasy?</h2>" + _p(2, "In a 2023 survey, 62% of users with oily skin reported greasiness.")
            + "<h2>Gel vs cream sunscreen for oily skin</h2>" + _p(2) + "<h2>Mineral vs chemical sunscreen for oily skin</h2>" + _p(2)
            + "<h2>Does sunscreen cause acne on oily skin?</h2>" + _p(2) + "<h2>How to reapply sunscreen over makeup</h2>" + _p(2)
            + "<h3>Is niacinamide sunscreen good for oily skin?</h3>" + _p(1),
            jsonld=[_faq_ld(["Can I skip moisturizer if I use sunscreen on oily skin?", "Which SPF is best for oily skin?"])],
        ),
        "guides/sunscreen-oily-skin-mistakes/index.html": _page(
            "Sunscreen Mistakes People With Oily Skin Make", "Sunscreen mistakes for oily skin",
            "<h2>Using too little sunscreen</h2>" + _p(2) + "<h2>Skipping sunscreen indoors</h2>" + _p(2)
            + "<h2>Why does sunscreen make oily skin greasy?</h2>" + _p(1),
        ),
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # pragma: no cover - silence test output
        pass


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(directory: Path) -> tuple[_Server, str]:
    """Serve ``directory`` on a free localhost port. Returns (server, base_url)."""
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = _Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def serve_fixture_site(tmp: Path) -> tuple[_Server, str]:
    """Build + serve the fixture store. The site must know its own base URL, so bind first."""
    site_dir = tmp / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    server, base = serve(site_dir)
    build_site(site_dir, base)
    return server, base


if __name__ == "__main__":  # manual: python -m tests.fixture_site
    import tempfile
    import time

    tmp = Path(tempfile.mkdtemp())
    srv, url = serve_fixture_site(tmp)
    print(f"Fixture site running at {url} (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.shutdown()
