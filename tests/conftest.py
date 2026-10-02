from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seo_fanout.config import Settings  # noqa: E402
from seo_fanout.db import Database  # noqa: E402
from seo_fanout.models import OrganicResult, SerpData  # noqa: E402
from seo_fanout.providers.serp import BaseSerpProvider  # noqa: E402
from tests.fixture_site import build_competitor_site, serve, serve_fixture_site  # noqa: E402


@pytest.fixture(scope="session")
def fixture_site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fixture")
    server, base = serve_fixture_site(tmp)
    yield base
    server.shutdown()


@pytest.fixture(scope="session")
def competitor_site(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("competitor")
    build_competitor_site(tmp)
    server, base = serve(tmp)
    yield base
    server.shutdown()


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.db_path = tmp_path / "test.sqlite3"
    s.output_dir = tmp_path / "output"
    s.log_dir = tmp_path / "logs"
    s.crawl.crawl_delay = 0.0
    s.crawl.max_retries = 1
    s.crawl.timeout = 5.0
    s.use_sentence_transformers = False
    s.llm_enabled = False
    s.free_sources_enabled = False  # the internet is not used in tests; see test_free_sources.py
    return s


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "unit.sqlite3")


class FakeSerpProvider(BaseSerpProvider):
    """TEST-ONLY provider returning canned SERP data (the app never fabricates SERPs)."""

    name = "fake"

    def __init__(self, db: Database, competitor_base: str):
        super().__init__(db)
        self.competitor_base = competitor_base

    def _search_raw(self, query, gl, hl, num, google_domain):
        cb = self.competitor_base
        return {
            "organic": [
                {"position": 1, "link": f"{cb}/oily-skin-sunscreen/", "title": "10 Best Sunscreens for Oily Skin", "snippet": "..."},
                {"position": 2, "link": f"{cb}/guides/sunscreen-oily-skin-mistakes/", "title": "Sunscreen mistakes", "snippet": "..."},
                {"position": 3, "link": "https://www.reddit.com/r/IndianSkincareAddicts/comments/abc/sunscreen_for_oily_skin/",
                 "title": "Which sunscreen doesn't make oily skin greasy? : r/IndianSkincareAddicts", "snippet": "Every sunscreen leaves me shiny. Is gel better?"},
                {"position": 4, "link": "https://www.quora.com/Which-sunscreen-is-best-for-oily-skin",
                 "title": "Which sunscreen doesn't make oily skin greasy? - Quora", "snippet": "Why does sunscreen make my face oily?"},
                {"position": 5, "link": "https://forum.skincaretalk.example/threads/oily-skin-sunscreen.12/",
                 "title": "Does sunscreen cause acne on oily skin?", "snippet": "Can I skip moisturizer if my sunscreen is hydrating?"},
            ],
            "peopleAlsoAsk": [
                {"question": "Which sunscreen is best for oily skin?", "snippet": "Gel-based...", "link": f"{cb}/oily-skin-sunscreen/"},
                {"question": "Why does sunscreen make my skin oily?", "snippet": "", "link": ""},
                {"question": "Is SPF 50 too much for oily skin?", "snippet": "", "link": ""},
                {"question": "Should I use moisturizer before sunscreen on oily skin?", "snippet": "", "link": ""},
            ],
            "relatedSearches": [
                {"query": "gel sunscreen for oily skin"}, {"query": "mineral sunscreen for oily skin"},
                {"query": "sunscreen for oily acne prone skin"}, {"query": "matte sunscreen for oily skin"},
                {"query": f"best sunscreen for oily skin {query.split()[0]}"},
            ],
        }

    def _autocomplete_raw(self, query, gl, hl):
        return {"suggestions": [{"value": "sunscreen for oily skin in summer"}, {"value": "sunscreen for oily skin for men"},
                                {"value": "sunscreen for oily skin and acne"}]}

    def _normalize(self, query, raw):
        from seo_fanout.providers.serp import SerperProvider

        return SerperProvider._normalize(self, query, raw)  # type: ignore[arg-type]

    def _parse_autocomplete(self, raw):
        return [s["value"] for s in raw["suggestions"]]


@pytest.fixture
def fake_serp_factory():
    return FakeSerpProvider


__all__ = ["FakeSerpProvider", "OrganicResult", "SerpData"]
