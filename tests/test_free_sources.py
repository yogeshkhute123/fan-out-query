"""Key-free mode: autocomplete, Wikipedia, Reddit public search and user-supplied competitors.

A local stub server stands in for the public endpoints (tests never touch the internet).
"""

import http.server
import json
import threading
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from seo_fanout.pipeline import ResearchInput, ResearchPipeline
from seo_fanout.providers.free_sources import AutocompleteEngine, RedditPublicSource, WikipediaSource
from seo_fanout.research.free_research import build_probes, demand_proxy

SUGGESTIONS = {
    "sunscreen for oily skin": ["sunscreen for oily skin", "sunscreen for oily skin and acne",
                                "sunscreen for oily skin in summer", "gel sunscreen for oily skin"],
    "why sunscreen for oily skin": ["why does sunscreen make my face oily", "why sunscreen for oily skin is important"],
    "how to sunscreen for oily skin": ["how to apply sunscreen on oily skin without looking greasy"],
    "can sunscreen for oily skin": ["can sunscreen cause pimples on oily skin"],
    "sunscreen for oily skin without": ["sunscreen for oily skin without white cast"],
    "sunscreen for oily skin vs": ["sunscreen for oily skin vs moisturizer"],
    "best sunscreen for oily skin": ["best sunscreen for oily skin in india", "best sunscreen for oily skin under 500"],
    "sunscreen for oily skin m": ["sunscreen for oily skin men", "sunscreen for oily skin matte finish"],
}


class StubHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parts = urlsplit(self.path)
        qs = {k: v[0] for k, v in parse_qs(parts.query).items()}
        if parts.path in ("/google", "/bing", "/ddg"):
            q = qs.get("q") or qs.get("query", "")
            sugg = SUGGESTIONS.get(q, [])
            if parts.path == "/bing":
                sugg = sugg[:2]  # engines disagree a little
            self._json([q, sugg])
        elif parts.path == "/w/api.php":
            if qs.get("action") == "query":
                self._json({"query": {"search": [{"title": "Sunscreen"}, {"title": "Zinc oxide"}, {"title": "Sun protection factor"}]}})
            else:
                self._json({"parse": {"sections": [{"line": "Health effects", "toclevel": 1}, {"line": "Active ingredients", "toclevel": 1},
                                                   {"line": "References", "toclevel": 1}]}})
        elif parts.path == "/search.json":
            self._json({"data": {"children": [
                {"data": {"title": "Why does every sunscreen make my oily skin greasy?", "permalink": "/r/skincare/comments/1/a/",
                          "selftext": "Is gel sunscreen better for oily skin?", "score": 50, "num_comments": 40, "subreddit": "skincare"}},
                {"data": {"title": "Which sunscreen doesn't break out oily skin?", "permalink": "/r/IndianSkincare/comments/2/b/",
                          "selftext": "", "score": 20, "num_comments": 15, "subreddit": "IndianSkincare"}},
            ]}})
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def stub(monkeypatch):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(AutocompleteEngine, "ENDPOINTS", {"google": f"{base}/google", "bing": f"{base}/bing",
                                                          "duckduckgo": f"{base}/ddg"})
    monkeypatch.setattr(WikipediaSource, "API", base + "/w/api.php")
    monkeypatch.setattr(RedditPublicSource, "SEARCH", base + "/search.json")
    yield base
    server.shutdown()


def test_probes():
    basic = build_probes("Sunscreen for oily skin", "basic")
    full = build_probes("Sunscreen for oily skin", "full")
    assert "why sunscreen for oily skin" in basic and "sunscreen for oily skin vs" in basic
    assert "sunscreen for oily skin z" in full and len(full) == len(basic) + 26


def test_demand_proxy_is_bounded_and_null_without_data():
    assert demand_proxy(None, 3) is None
    assert demand_proxy({"engines": {"google", "bing", "duckduckgo"}, "best_rank": 1}, 3) == 1.0
    assert 0 < demand_proxy({"engines": {"google"}, "best_rank": 8}, 3) < 0.5


def test_keyless_pipeline(stub, fixture_site, competitor_site, settings):
    settings.free_sources_enabled = True
    settings.suggest_delay = 0
    settings.competitor_urls = [f"{competitor_site}/oily-skin-sunscreen/"]
    pipe = ResearchPipeline(settings)
    try:
        result, paths = pipe.run(ResearchInput(seed="sunscreen for oily skin", base_url=fixture_site, country="India",
                                               language="English", num_queries=100))
    finally:
        pipe.close()
    fs = result.stats["free_sources"]
    assert set(fs["engines_used"]) == {"google", "bing", "duckduckgo"}
    assert fs["wikipedia_article"] == "Sunscreen"
    origins = set().union(*(c.origins for c in result.candidates))
    assert {"autocomplete", "bing_suggest", "ddg_suggest", "wikipedia", "reddit", "competitors"} <= origins
    # user-supplied competitor analysed without any SERP API
    assert result.competitors.available and result.competitors.pages[0].fetched
    # no fabricated volume, but an explicitly estimated demand signal for autocomplete queries
    assert all(c.metrics.search_volume is None for c in result.final)
    ac = [c for c in result.final if c.origins & {"autocomplete", "bing_suggest", "ddg_suggest"}]
    assert ac and all(c.demand_proxy is not None and "search_demand_proxy" in c.score_breakdown for c in ac)
    # queries seen by all three engines rank above template-only ideas on average
    tpl = [c.score for c in result.final if c.origins == {"template"}]
    multi = [c.score for c in result.final if len(c.origins & {"autocomplete", "bing_suggest", "ddg_suggest"}) >= 2]
    assert multi and tpl and sum(multi) / len(multi) > sum(tpl) / len(tpl)
    df = pd.read_csv(paths["csv"])
    assert df["demand_signal_estimated"].notna().any()
    assert (df["source_bing_ddg_autocomplete"] == "YES").any()
    assert len(result.final) > 34  # more than the keyless baseline without free sources


def test_blocked_engine_fails_softly(db):
    engine = AutocompleteEngine("google", db, "test-agent", delay=0)
    engine.ENDPOINTS = {"google": "http://127.0.0.1:9/never"}  # closed port
    engine.http.client.timeout = 0.5
    for _ in range(6):
        assert engine.suggest("x", "in", "en") == []
    assert engine.consecutive_failures >= 5
