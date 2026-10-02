from seo_fanout.analysis.embeddings import SemanticModel
from seo_fanout.models import (
    Action, Coverage, DataLabel, Intent, KeywordMetrics, PageInventory, PageType, QueryCandidate,
)
from seo_fanout.queries.dedup import deduplicate
from seo_fanout.queries.generator import CandidatePool, observed_variants, split_seed
from seo_fanout.queries.quality import apply_quality_control, compute_topical_relevance
from seo_fanout.queries.scoring import score_candidates
from seo_fanout.queries.selection import select_final
from seo_fanout.queries.url_recommender import SiteArchitecture


def _model(texts):
    return SemanticModel(use_transformers=False).fit(texts)


def _cand(q, origins=("template",), label=DataLabel.TEMPLATE):
    return QueryCandidate(query=q, normalized=q, origins=set(origins), label=label)


def test_split_seed():
    p = split_seed("Sunscreen for Oily Skin")
    assert (p.head, p.qualifier) == ("sunscreen", "oily skin")
    assert split_seed("protein powder").qualifier is None


def test_observed_variants_only_noun_phrases():
    p = split_seed("sunscreen for oily skin")
    texts = [("page1", "Gel sunscreen picks"), ("page2", "Is gel sunscreen good?"), ("page1", "mineral sunscreen guide"),
             ("comp1", "Mineral sunscreen vs chemical"), ("page1", "buy sunscreen online"), ("page2", "buy sunscreen now"),
             ("page1", "The Complete Sunscreen Guide"), ("page1", "complete sunscreen"),  # one source only
             ("page1", "tinted sunscreen"), ("page1", "Tinted sunscreen tips")]  # one source only
    variants = observed_variants(p, texts)
    assert set(variants) == {"gel sunscreen", "mineral sunscreen"}


def test_dedup_removes_variants_but_keeps_distinct_needs():
    queries = [
        "best sunscreen", "best sunscreen india", "best sunscreen in india", "best sunscreen online",
        "best sunscreens", "Best sunscreen?", "sunscreen best",
        "best sunscreen for oily skin", "why does sunscreen make oily skin oily",
        "how often should sunscreen be reapplied", "can oily skin use moisturizer with sunscreen",
    ]
    cands = [_cand(q) for q in queries]
    pool = CandidatePool()
    for q in queries:
        pool.add(q, "template")
    cands = list(pool.items.values())
    model = _model(queries)
    out, stats = deduplicate(cands, model, country="India")
    kept = {c.query for c in out}
    assert "best sunscreen for oily skin" in kept
    assert "why does sunscreen make oily skin oily" in kept
    assert "how often should sunscreen be reapplied" in kept
    assert "can oily skin use moisturizer with sunscreen" in kept
    best_variants = {"best sunscreen", "best sunscreen india", "best sunscreen in india", "best sunscreen online",
                     "best sunscreens", "sunscreen best"}
    assert len(kept & best_variants) == 1
    assert stats["removed"] >= 5


def test_dedup_prefers_real_evidence():
    a = _cand("sunscreen for oily skin reviews")
    b = _cand("sunscreen for oily skin review", origins=("paa",), label=DataLabel.REAL)
    out, _ = deduplicate([a, b], _model([a.query, b.query]))
    assert len(out) == 1 and out[0].label == DataLabel.REAL and out[0].origins == {"paa", "template"}


def test_quality_control_rejects_junk_and_unrelated():
    seed = "sunscreen for oily skin"
    cands = [_cand(q) for q in ("sunscreen", "best laptop for gaming", "check https://x.com sunscreen",
                                 "is gel sunscreen good for oily skin")]
    compute_topical_relevance(cands, seed, _model([c.query for c in cands] + [seed]))
    kept, rejected = apply_quality_control(cands, seed)
    assert [c.query for c in kept] == ["is gel sunscreen good for oily skin"]
    assert {c.rejected_reason.split(":")[0] for c in rejected} >= {"meaningless", "unrelated to the seed topic"}


def test_scoring_never_imputes_volume():
    a = _cand("gel sunscreen for oily skin", origins=("paa", "reddit"), label=DataLabel.REAL)
    b = _cand("matte sunscreen for oily skin")
    for c in (a, b):
        c.intent, c.intent_confidence, c.coverage, c.topical_relevance = Intent.COMMERCIAL, 0.8, Coverage.CONTENT_GAP, 0.8
        c.metrics = KeywordMetrics(keyword=c.query)  # unavailable
    score_candidates([a, b])
    assert "search_volume" not in a.score_breakdown
    assert a.metrics.search_volume is None
    assert a.score > b.score  # real multi-source evidence beats a template
    assert a.confidence == "HIGH" and b.confidence == "LOW"


def test_selection_is_balanced_and_honest():
    cands = []
    for i in range(30):
        c = _cand(f"best sunscreen variant number {i}")
        c.cluster, c.intent, c.query_type, c.score = "same", Intent.COMMERCIAL, "best-of / list", 90 - i * 0.1
        cands.append(c)
    for i, q in enumerate(["why sunscreen greasy", "gel vs cream sunscreen", "how to apply sunscreen", "sunscreen price"]):
        c = _cand(q)
        c.cluster, c.intent, c.query_type, c.score = f"c{i}", list(Intent)[i], f"t{i}", 50
        cands.append(c)
    final, info = select_final(cands, 20, _model([c.query for c in cands]))
    same = sum(1 for c in final if c.cluster == "same")
    assert same < 0.8 * len(final)
    assert all(c in final for c in cands[30:])


def test_url_recommender_follows_site_structure():
    inv = [
        PageInventory(url="https://e.com/blog/sunscreen-guide/", page_type=PageType.BLOG, primary_topic="sunscreen guide"),
        PageInventory(url="https://e.com/blog/acne/", page_type=PageType.BLOG, primary_topic="acne"),
        PageInventory(url="https://e.com/collections/sunscreen/", page_type=PageType.COLLECTION, primary_topic="sunscreen"),
        PageInventory(url="https://e.com/products/gel/", page_type=PageType.PRODUCT, primary_topic="gel"),
    ]
    arch = SiteArchitecture(inv, "https://e.com")
    url, _ = arch.recommend(Action.NEW_PAGE, "why does sunscreen make oily skin greasy", None)
    assert url == "https://e.com/blog/why-sunscreen-make-oily-skin-greasy/"
    url, _ = arch.recommend(Action.CATEGORY_PAGE, "matte sunscreen", None)
    assert url.startswith("https://e.com/collections/")
    url, _ = arch.recommend(Action.EXISTING_PAGE_OPTIMIZATION, "x", "https://e.com/blog/acne/")
    assert url == "https://e.com/blog/acne/"
