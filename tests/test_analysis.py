from seo_fanout.analysis.cannibalization import detect_cannibalization
from seo_fanout.analysis.embeddings import SemanticModel
from seo_fanout.analysis.intent import classify_intent
from seo_fanout.analysis.inventory import build_inventory
from seo_fanout.crawler.parser import parse_html
from seo_fanout.models import Intent, PageType, Risk
from seo_fanout.textutil import canonical_text, key_phrases, slugify, strip_brand_suffix


def _page(url, title, h1, h2s, body, schema=""):
    h2 = "".join(f"<h2>{h}</h2><p>{body}</p>" for h in h2s)
    return parse_html(url, f"<html><head><title>{title}</title>{schema}</head><body><h1>{h1}</h1>{h2}</body></html>", "e.com")


def test_text_helpers():
    assert strip_brand_suffix("Best Sunscreen for Oily Skin | GlowSkin") == "Best Sunscreen for Oily Skin"
    assert strip_brand_suffix("Blog | GlowSkin") == "Blog"
    assert canonical_text("Top Sunscreens") == canonical_text("best sunscreen")
    assert slugify("Why does sunscreen make oily skin greasy?") == "why-sunscreen-make-oily-skin-greasy"
    assert "gel sunscreen" in " ".join(key_phrases("Gel sunscreen is light. Gel sunscreen absorbs. Gel sunscreen works."))


def test_intent_classifier():
    cases = {
        "buy sunscreen online": Intent.TRANSACTIONAL,
        "best sunscreen for oily skin": Intent.COMMERCIAL,
        "gel vs cream sunscreen": Intent.COMPARISON,
        "why does sunscreen make oily skin greasy": Intent.PROBLEM_SOLVING,
        "what is spf": Intent.INFORMATIONAL,
        "sunscreen price": Intent.TRANSACTIONAL_INVESTIGATION,
        "dermatologist near me": Intent.LOCAL,
        "how to get the best results from sunscreen": Intent.INFORMATIONAL,
    }
    for q, expected in cases.items():
        assert classify_intent(q).intent == expected, q


def test_cannibalization_same_need_vs_shared_keyword():
    body = "Oily skin sunscreen gel matte finish SPF non comedogenic oil free texture lightweight " * 20
    pages = [
        _page("https://e.com/blog/best-sunscreen-for-oily-skin", "Best Sunscreen for Oily Skin | B",
              "Best Sunscreen for Oily Skin", ["Gel picks", "Matte picks"], body),
        _page("https://e.com/blog/top-sunscreens-for-oily-skin", "Top Sunscreens for Oily Skin | B",
              "Top Sunscreens for Oily Skin", ["Gel sunscreen picks", "Matte sunscreen picks"], body),
        # shares the keyword "sunscreen" but serves a different need/intent
        _page("https://e.com/blog/why-sunscreen-makes-skin-greasy", "Why Sunscreen Makes Skin Greasy | B",
              "Why does sunscreen make oily skin greasy?", ["Causes of greasiness", "How to fix it"],
              "Greasy residue happens because of emollients and silicones in the formula " * 20),
        _page("https://e.com/blog/vitamin-c", "Vitamin C Serum Benefits | B", "Vitamin C Serum Benefits",
              ["Brightening"], "Vitamin C brightens skin and fades spots " * 20),
    ]
    for p in pages:
        p.page_type = PageType.BLOG
    inv = build_inventory(pages)
    model = SemanticModel(use_transformers=False).fit([p.title + " " + p.main_content for p in pages])
    pairs, _ = detect_cannibalization(pages, inv, model)
    by_pair = {frozenset((p.url_a, p.url_b)): p for p in pairs}
    best_top = by_pair[frozenset(("https://e.com/blog/best-sunscreen-for-oily-skin", "https://e.com/blog/top-sunscreens-for-oily-skin"))]
    assert best_top.risk == Risk.HIGH
    assert best_top.stronger_url and best_top.recommendation
    greasy = by_pair.get(frozenset(("https://e.com/blog/best-sunscreen-for-oily-skin", "https://e.com/blog/why-sunscreen-makes-skin-greasy")))
    assert greasy is None or greasy.risk in (Risk.LOW, Risk.NONE)
    assert not any("vitamin-c" in p.url_a + p.url_b for p in pairs if p.risk != Risk.LOW)


def test_inventory_fields():
    p = _page("https://e.com/blog/how-to-apply-sunscreen", "How to Apply Sunscreen | B", "How to Apply Sunscreen",
              ["How much sunscreen should you apply?", "Reapplying over makeup"], "apply sunscreen evenly " * 30)
    p.page_type = PageType.BLOG
    inv = build_inventory([p])[0]
    assert inv.primary_topic == "how to apply sunscreen"
    assert inv.content_type == "How-to guide"
    assert "How much sunscreen should you apply?" in inv.questions_answered
    assert inv.funnel_stage == "TOFU"
