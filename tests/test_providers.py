import logging

import pytest

from seo_fanout.config import Settings
from seo_fanout.logging_setup import SecretRedactingFilter
from seo_fanout.providers.base import ProviderUnavailable, extract_json
from seo_fanout.providers.google_ads import GoogleAdsKeywordPlanner
from seo_fanout.providers.locale import country_info, language_info
from seo_fanout.providers.openai import OpenAIProvider
from seo_fanout.providers.reddit import RedditProvider
from seo_fanout.providers.serp import get_serp_provider


def test_optional_providers_raise_unavailable_instead_of_crashing(db):
    s = Settings()
    with pytest.raises(ProviderUnavailable):
        get_serp_provider(s, db)
    with pytest.raises(ProviderUnavailable):
        GoogleAdsKeywordPlanner(s, db)
    with pytest.raises(ProviderUnavailable):
        RedditProvider(s, db)
    with pytest.raises(ProviderUnavailable):
        OpenAIProvider(None, "x", db)


def test_locale_mapping():
    assert country_info("India")[:2] == ("in", 2356)
    assert country_info("us")[:2] == ("us", 2840)
    assert language_info("English") == ("en", 1000)
    assert country_info("Atlantis")[1] is None


def test_extract_json_from_llm_text():
    assert extract_json('```json\n{"a": [1]}\n```') == {"a": [1]}
    assert extract_json('Sure! {"queries": ["x"]} hope that helps') == {"queries": ["x"]}
    assert extract_json("no json here") is None


def test_secret_redaction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecretvalue123456")
    f = SecretRedactingFilter()
    rec = logging.LogRecord("x", logging.INFO, "", 0, "key=%s url=https://a?api_key=abc123", ("sk-supersecretvalue123456",), None)
    f.filter(rec)
    assert "supersecret" not in rec.msg and "abc123" not in rec.msg


def test_cache_roundtrip(db):
    db.cache_set("ns", "k", {"v": 1})
    assert db.cache_get("ns", "k") == {"v": 1}
    assert db.cache_get("ns", "other") is None
