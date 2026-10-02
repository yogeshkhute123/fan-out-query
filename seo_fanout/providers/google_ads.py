"""Google Ads Keyword Planner provider (KeywordPlanIdeaService.GenerateKeywordHistoricalMetrics).

This is the ONLY source of search volume / competition / bid data in the tool.
If it is not configured or a keyword has no data, metrics stay ``None`` – the
tool never estimates or invents a volume.
"""

from __future__ import annotations

import time

from ..config import Settings
from ..db import Database
from ..logging_setup import get_logger
from ..models import DataLabel, KeywordMetrics
from ..textutil import normalize_query
from .base import ProviderAuthError, ProviderUnavailable
from .locale import country_info, language_info

log = get_logger("google_ads")

BATCH_SIZE = 500  # API accepts up to 10k; smaller batches keep failures cheap
SOURCE = "google_ads"


class GoogleAdsKeywordPlanner:
    def __init__(self, settings: Settings, db: Database):
        if not settings.google_ads_configured:
            raise ProviderUnavailable("Google Ads credentials not configured")
        try:
            from google.ads.googleads.client import GoogleAdsClient  # type: ignore
        except ImportError as exc:
            raise ProviderUnavailable("google-ads package not installed (pip install google-ads)") from exc
        config = {
            "developer_token": settings.google_ads_developer_token,
            "client_id": settings.google_ads_client_id,
            "client_secret": settings.google_ads_client_secret,
            "refresh_token": settings.google_ads_refresh_token,
            "use_proto_plus": True,
        }
        if settings.google_ads_login_customer_id:
            config["login_customer_id"] = settings.google_ads_login_customer_id
        try:
            self.client = GoogleAdsClient.load_from_dict(config)
        except Exception as exc:
            raise ProviderAuthError(f"Google Ads client init failed: {type(exc).__name__}") from exc
        self.customer_id = settings.google_ads_customer_id
        self.db = db
        self.ttl_days = settings.cache_ttl_days
        self.calls = 0

    # ------------------------------------------------------------------
    def _resolve_geo(self, country: str) -> int | None:
        gl, geo, _ = country_info(country)
        if geo:
            return geo
        try:
            svc = self.client.get_service("GeoTargetConstantService")
            req = self.client.get_type("SuggestGeoTargetConstantsRequest")
            req.locale = "en"
            req.location_names.names.append(country)
            resp = svc.suggest_geo_target_constants(request=req)
            for suggestion in resp.geo_target_constant_suggestions:
                gtc = suggestion.geo_target_constant
                if gtc.target_type == "Country":
                    return int(gtc.id)
        except Exception as exc:
            log.warning("Geo target lookup failed for %s: %s", country, exc)
        return None

    def _call(self, keywords: list[str], geo_id: int | None, lang_id: int | None):
        svc = self.client.get_service("KeywordPlanIdeaService")
        req = self.client.get_type("GenerateKeywordHistoricalMetricsRequest")
        req.customer_id = self.customer_id
        req.keywords.extend(keywords)
        if geo_id:
            req.geo_target_constants.append(f"geoTargetConstants/{geo_id}")
        if lang_id:
            req.language = f"languageConstants/{lang_id}"
        req.keyword_plan_network = self.client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
        return svc.generate_keyword_historical_metrics(request=req)

    def _call_with_backoff(self, keywords, geo_id, lang_id):
        from google.ads.googleads.errors import GoogleAdsException  # type: ignore

        delay = 5.0
        for attempt in range(5):
            try:
                self.calls += 1
                return self._call(keywords, geo_id, lang_id)
            except GoogleAdsException as exc:
                codes = " ".join(str(e.error_code) for e in exc.failure.errors)
                if "RESOURCE_EXHAUSTED" in codes or "RESOURCE_TEMPORARILY_EXHAUSTED" in codes or "rate" in codes.lower():
                    log.warning("Google Ads rate limit – sleeping %.0fs (attempt %d)", delay, attempt + 1)
                    time.sleep(delay)
                    delay *= 2
                    continue
                if "AUTHENTICATION" in codes or "AUTHORIZATION" in codes or "DEVELOPER_TOKEN" in codes:
                    raise ProviderAuthError(f"Google Ads auth error: {codes}") from exc
                raise
            except Exception as exc:  # transport errors
                if attempt == 4:
                    raise
                log.warning("Google Ads transient error (%s) – retrying in %.0fs", type(exc).__name__, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("Google Ads retries exhausted")

    @staticmethod
    def _micros(value) -> float | None:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return None
        return round(v / 1_000_000, 2) if v else None

    def fetch(self, keywords: list[str], country: str, language: str) -> dict[str, KeywordMetrics]:
        """Return metrics keyed by normalized keyword. Missing keywords get no entry."""
        results: dict[str, KeywordMetrics] = {}
        pending: list[str] = []
        for kw in dict.fromkeys(keywords):
            norm = normalize_query(kw)
            cached = self.db.get_keyword_metrics(norm, country.lower(), language.lower(), SOURCE, self.ttl_days)
            if cached is not None:
                results[norm] = KeywordMetrics(**cached)
            else:
                pending.append(kw)
        if not pending:
            return results

        geo_id = self._resolve_geo(country)
        _, lang_id = language_info(language)
        if geo_id is None:
            log.warning("No Google Ads geo target for '%s' – requesting worldwide metrics", country)

        for i in range(0, len(pending), BATCH_SIZE):
            batch = [k[:80] for k in pending[i : i + BATCH_SIZE]]
            log.info("Google Ads historical metrics: batch %d (%d keywords)", i // BATCH_SIZE + 1, len(batch))
            response = self._call_with_backoff(batch, geo_id, lang_id)
            returned: set[str] = set()
            for result in response.results:
                m = result.keyword_metrics
                monthly = [
                    {"year": int(v.year), "month": getattr(v.month, "name", str(v.month)), "searches": int(v.monthly_searches)}
                    for v in getattr(m, "monthly_search_volumes", [])
                ]
                comp = getattr(m.competition, "name", str(m.competition)) if m else None
                metrics = KeywordMetrics(
                    keyword=result.text,
                    search_volume=int(m.avg_monthly_searches) if m and m.avg_monthly_searches is not None else None,
                    competition=comp if comp not in ("UNSPECIFIED", "UNKNOWN") else None,
                    competition_index=int(m.competition_index) if m and m.competition_index is not None else None,
                    cpc_low=self._micros(getattr(m, "low_top_of_page_bid_micros", None)),
                    cpc_high=self._micros(getattr(m, "high_top_of_page_bid_micros", None)),
                    monthly=monthly,
                    source=SOURCE,
                    label=DataLabel.REAL,
                    note="Google Ads Keyword Planner historical metrics (cpc = top-of-page bid range)",
                )
                keys = {normalize_query(result.text)} | {normalize_query(v) for v in result.close_variants}
                for key in keys:
                    results[key] = metrics.model_copy(update={"keyword": key})
                    returned.add(key)
                    self.db.save_keyword_metrics(key, country.lower(), language.lower(), SOURCE, metrics.model_dump(mode="json"))
            log.info("Google Ads returned metrics for %d/%d keywords", len(returned & {normalize_query(b) for b in batch}), len(batch))
        return results
