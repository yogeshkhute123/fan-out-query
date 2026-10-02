"""Application configuration.

Every credential is read from environment variables (optionally loaded from a
local ``.env`` file). Nothing secret is ever hard-coded or written to logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))  # type: ignore[arg-type]
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))  # type: ignore[arg-type]
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# Names of every environment variable that holds a secret. The logging filter
# uses this list to redact values that accidentally reach a log message.
SECRET_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PERPLEXITY_API_KEY",
    "GOOGLE_ADS_DEVELOPER_TOKEN",
    "GOOGLE_ADS_CLIENT_ID",
    "GOOGLE_ADS_CLIENT_SECRET",
    "GOOGLE_ADS_REFRESH_TOKEN",
    "SERP_API_KEY",
    "SERPER_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
)


@dataclass
class CrawlSettings:
    user_agent: str = "SEOFanoutResearchBot/1.0 (+responsible crawler; contact site owner)"
    crawl_delay: float = 1.0
    max_pages: int = 300
    timeout: float = 20.0
    max_retries: int = 3
    max_sitemap_urls: int = 5000
    max_sitemaps: int = 50
    respect_robots: bool = True
    follow_internal_links: bool = True
    max_depth: int = 6


@dataclass
class Settings:
    # --- API credentials (all optional) ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    perplexity_api_key: str | None = None
    perplexity_model: str = "sonar"

    serp_provider: str = "serpapi"  # serpapi | serper
    serp_api_key: str | None = None
    serper_api_key: str | None = None

    google_ads_developer_token: str | None = None
    google_ads_client_id: str | None = None
    google_ads_client_secret: str | None = None
    google_ads_refresh_token: str | None = None
    google_ads_customer_id: str | None = None
    google_ads_login_customer_id: str | None = None

    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "seo-fanout-research/1.0"

    # --- behaviour ---
    use_sentence_transformers: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    serp_results_per_query: int = 10
    max_serp_queries: int = 14
    max_competitor_pages: int = 8
    llm_enabled: bool = True
    cache_ttl_days: int = 30

    # --- paths ---
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "fanout.sqlite3")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    crawl: CrawlSettings = field(default_factory=CrawlSettings)

    # ------------------------------------------------------------------
    @property
    def google_ads_configured(self) -> bool:
        return all(
            [
                self.google_ads_developer_token,
                self.google_ads_client_id,
                self.google_ads_client_secret,
                self.google_ads_refresh_token,
                self.google_ads_customer_id,
            ]
        )

    @property
    def serp_configured(self) -> bool:
        if self.serp_provider == "serper":
            return bool(self.serper_api_key)
        return bool(self.serp_api_key)

    @property
    def reddit_configured(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)

    def provider_status(self) -> dict[str, bool]:
        return {
            "SERP provider": self.serp_configured,
            "Google Ads Keyword Planner": self.google_ads_configured,
            "OpenAI": bool(self.openai_api_key) and self.llm_enabled,
            "Anthropic": bool(self.anthropic_api_key) and self.llm_enabled,
            "Perplexity": bool(self.perplexity_api_key) and self.llm_enabled,
            "Reddit API": self.reddit_configured,
        }


def load_settings(env_file: str | os.PathLike | None = None) -> Settings:
    """Load settings from the environment (and ``.env`` when present)."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    crawl = CrawlSettings(
        user_agent=_env("CRAWL_USER_AGENT", CrawlSettings.user_agent),  # type: ignore[arg-type]
        crawl_delay=_env_float("CRAWL_DELAY", CrawlSettings.crawl_delay),
        max_pages=_env_int("CRAWL_MAX_PAGES", CrawlSettings.max_pages),
        timeout=_env_float("CRAWL_TIMEOUT", CrawlSettings.timeout),
        max_retries=_env_int("CRAWL_MAX_RETRIES", CrawlSettings.max_retries),
        respect_robots=_env_bool("CRAWL_RESPECT_ROBOTS", True),
    )

    settings = Settings(
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_model=_env("OPENAI_MODEL", Settings.openai_model),  # type: ignore[arg-type]
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        anthropic_model=_env("ANTHROPIC_MODEL", Settings.anthropic_model),  # type: ignore[arg-type]
        perplexity_api_key=_env("PERPLEXITY_API_KEY"),
        perplexity_model=_env("PERPLEXITY_MODEL", Settings.perplexity_model),  # type: ignore[arg-type]
        serp_provider=(_env("SERP_PROVIDER", "serpapi") or "serpapi").lower(),
        serp_api_key=_env("SERP_API_KEY"),
        serper_api_key=_env("SERPER_API_KEY"),
        google_ads_developer_token=_env("GOOGLE_ADS_DEVELOPER_TOKEN"),
        google_ads_client_id=_env("GOOGLE_ADS_CLIENT_ID"),
        google_ads_client_secret=_env("GOOGLE_ADS_CLIENT_SECRET"),
        google_ads_refresh_token=_env("GOOGLE_ADS_REFRESH_TOKEN"),
        google_ads_customer_id=(_env("GOOGLE_ADS_CUSTOMER_ID") or "").replace("-", "") or None,
        google_ads_login_customer_id=(_env("GOOGLE_ADS_LOGIN_CUSTOMER_ID") or "").replace("-", "") or None,
        reddit_client_id=_env("REDDIT_CLIENT_ID"),
        reddit_client_secret=_env("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=_env("REDDIT_USER_AGENT", Settings.reddit_user_agent),  # type: ignore[arg-type]
        use_sentence_transformers=_env_bool("USE_SENTENCE_TRANSFORMERS", True),
        embedding_model=_env("EMBEDDING_MODEL", Settings.embedding_model),  # type: ignore[arg-type]
        max_serp_queries=_env_int("MAX_SERP_QUERIES", Settings.max_serp_queries),
        max_competitor_pages=_env_int("MAX_COMPETITOR_PAGES", Settings.max_competitor_pages),
        llm_enabled=_env_bool("LLM_ENABLED", True),
        crawl=crawl,
    )
    if _env("FANOUT_DB_PATH"):
        settings.db_path = Path(_env("FANOUT_DB_PATH"))  # type: ignore[arg-type]
    if _env("FANOUT_OUTPUT_DIR"):
        settings.output_dir = Path(_env("FANOUT_OUTPUT_DIR"))  # type: ignore[arg-type]
    if _env("FANOUT_LOG_DIR"):
        settings.log_dir = Path(_env("FANOUT_LOG_DIR"))  # type: ignore[arg-type]
    return settings
