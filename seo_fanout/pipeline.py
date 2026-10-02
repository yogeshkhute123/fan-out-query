"""End-to-end research pipeline (10 stages)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .analysis.cannibalization import detect_cannibalization
from .analysis.embeddings import SemanticModel
from .analysis.inventory import build_inventory, page_topic_text
from .analysis.topics import build_topic_map, cluster_queries, parent_topic_for
from .config import Settings
from .crawler.crawler import Crawler
from .crawler.urls import site_key
from .db import Database
from .export.csv_export import write_csv
from .export.excel import write_excel
from .export.reports import write_reports
from .logging_setup import get_logger
from .models import Coverage, PageData, QueryCandidate
from .providers.anthropic import AnthropicProvider
from .providers.base import ProviderError
from .providers.google_ads import GoogleAdsKeywordPlanner
from .providers.llm import LLMProvider
from .providers.openai import OpenAIProvider
from .providers.perplexity import PerplexityProvider
from .providers.free_sources import AutocompleteEngine, RedditPublicSource, WikipediaSource
from .providers.locale import language_info
from .providers.reddit import RedditProvider
from .providers.serp import BaseSerpProvider, get_serp_provider
from .queries.coverage import assess_coverage
from .queries.dedup import deduplicate
from .queries.generator import generate_candidates, seed_relevant_pages
from .queries.internal_links import find_internal_links
from .queries.quality import apply_quality_control, compute_topical_relevance
from .queries.scoring import compute_business_relevance, compute_serp_evidence, score_candidates
from .queries.selection import select_final
from .queries.url_recommender import SiteArchitecture
from .research.community import run_community_research
from .research.competitors import run_competitor_research
from .research.keyword_metrics import attach_keyword_metrics
from .research.llm_research import run_llm_research
from .research.paa import expand_paa
from .research.free_research import demand_proxy, run_autocomplete_research, run_wikipedia_research
from .research.serp_research import run_serp_research
from .results import ResearchResult

log = get_logger("pipeline")
TOTAL_STEPS = 10


@dataclass
class ResearchInput:
    seed: str
    base_url: str
    country: str = ""
    language: str = "English"
    num_queries: int = 100
    fresh_crawl: bool = False


def _noop(step: int, label: str) -> None:
    pass


def _reasoning(c: QueryCandidate) -> str:
    parts = [
        f"Intent {c.intent.value if c.intent else '?'} ({c.query_type}).",
        f"Coverage {c.coverage.value if c.coverage else '?'} → {c.action.value if c.action else '?'}.",
    ]
    real = sorted(o for o in c.origins if o not in ("template", "llm"))
    if real:
        parts.append("Observed in: " + ", ".join(real) + ".")
    if "llm" in c.origins:
        parts.append("Also suggested by an LLM (LLM_GENERATED).")
    if c.origins == {"template"}:
        parts.append("Pattern-generated (no external evidence) – validate demand before investing.")
    m = c.metrics
    if m is None or m.search_volume is None:
        parts.append(f"Search volume unavailable ({m.note if m else 'not requested'}); score excludes volume.")
    else:
        parts.append(f"Search volume {m.search_volume}/mo (Google Ads, REAL_DATA).")
    top = sorted(c.score_breakdown.items(), key=lambda kv: -kv[1])[:3]
    parts.append("Strongest signals: " + ", ".join(f"{k}={v:.2f}" for k, v in top) + ".")
    return " ".join(parts)


class ResearchPipeline:
    def __init__(self, settings: Settings, progress: Callable[[int, str], None] | None = None):
        self.settings = settings
        self.progress = progress or _noop
        self.db = Database(settings.db_path)
        self.notes: list[str] = []
        self.skipped_apis: list[str] = []

    # ------------------------------------------------------------ providers
    def _init_serp(self) -> BaseSerpProvider | None:
        try:
            return get_serp_provider(self.settings, self.db)
        except ProviderError as exc:
            self.skipped_apis.append("SERP API")
            log.info("SERP provider unavailable: %s", exc)
            return None

    def _init_llms(self) -> list[LLMProvider]:
        providers: list[LLMProvider] = []
        if not self.settings.llm_enabled:
            self.notes.append("LLM providers disabled (LLM_ENABLED=false)")
            return providers
        s = self.settings
        for label, factory in (
            ("Anthropic", lambda: AnthropicProvider(s.anthropic_api_key, s.anthropic_model, self.db)),
            ("OpenAI", lambda: OpenAIProvider(s.openai_api_key, s.openai_model, self.db)),
            ("Perplexity", lambda: PerplexityProvider(s.perplexity_api_key, s.perplexity_model, self.db)),
        ):
            try:
                providers.append(factory())
            except ProviderError as exc:
                log.info("LLM provider skipped: %s", exc)
                self.skipped_apis.append(label)
        return providers

    def _init_reddit(self) -> RedditProvider | RedditPublicSource | None:
        s = self.settings
        try:
            return RedditProvider(s, self.db)
        except ProviderError as exc:
            log.info("Reddit API unavailable: %s", exc)
        if s.free_sources_enabled and s.reddit_public_enabled:
            return RedditPublicSource(self.db, s.crawl.user_agent)
        self.skipped_apis.append("Reddit API")
        return None

    def _init_free_sources(self, language: str) -> tuple[list[AutocompleteEngine], WikipediaSource | None]:
        s = self.settings
        if not s.free_sources_enabled:
            self.notes.append("Key-free sources disabled (FREE_SOURCES_ENABLED=false / --no-free-sources)")
            return [], None
        engines = []
        for name in s.autocomplete_engines:
            try:
                engines.append(AutocompleteEngine(name, self.db, s.crawl.user_agent, s.suggest_delay))
            except ValueError as exc:
                self.notes.append(str(exc))
        wiki = WikipediaSource(self.db, s.crawl.user_agent, language_info(language)[0]) if s.wikipedia_enabled else None
        return engines, wiki

    def _init_ads(self) -> tuple[GoogleAdsKeywordPlanner | None, str]:
        try:
            return GoogleAdsKeywordPlanner(self.settings, self.db), ""
        except ProviderError as exc:
            reason = f"Google Ads Keyword Planner unavailable: {exc}"
            log.info(reason)
            self.skipped_apis.append("Google Ads Keyword Planner")
            return None, reason

    # ---------------------------------------------------------------- crawl
    def _crawl(self, website_id: int, base_url: str, fresh: bool) -> tuple[list[PageData], bool, dict]:
        info = self.db.website_info(website_id)
        if not fresh and info is not None and info["last_crawled_at"]:
            cached = [PageData(**p) for p in self.db.load_pages(website_id)]
            if cached:
                log.info("Reusing crawl from %s (%d pages)", info["last_crawled_at"], len(cached))
                return cached, True, {"reused_crawl_from": info["last_crawled_at"]}
        if fresh:
            self.db.clear_pages(website_id)
        crawler = Crawler(base_url, self.settings.crawl, self.db, website_id)
        try:
            # Fresh crawls must not be served from the HTTP cache.
            if fresh:
                original = crawler.fetcher.fetch
                crawler.fetcher.fetch = lambda url, use_cache=True: original(url, use_cache=False)  # type: ignore[method-assign]
            pages = crawler.crawl()
        finally:
            crawler.close()
        st = crawler.stats
        return pages, False, {
            "urls_discovered": st.discovered, "pages_fetched": st.fetched, "sitemap_urls": st.sitemap_urls,
            "sitemaps": len(st.sitemaps), "robots_skipped": st.skipped_robots, "duplicate_urls_skipped": st.duplicates,
            "failed": st.failed, "http_status": dict(st.status_counts), "robots_note": st.robots_note,
        }

    # ------------------------------------------------------------------ run
    def run(self, inp: ResearchInput) -> tuple[ResearchResult, dict[str, Path]]:
        s = self.settings
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        host = urlsplit(inp.base_url).hostname or inp.base_url
        domain = site_key(host)
        website_id = self.db.get_or_create_website(domain, inp.base_url)
        self.db.create_run(run_id, website_id, inp.seed, inp.country, inp.language, inp.num_queries)
        log.info("=== Research run %s: seed='%s' site=%s ===", run_id, inp.seed, inp.base_url)
        result = ResearchResult(run_id=run_id, seed=inp.seed, domain=domain, base_url=inp.base_url,
                                country=inp.country, language=inp.language, requested=inp.num_queries,
                                provider_status=s.provider_status())
        try:
            paths = self._run(inp, result, website_id)
            self.db.finish_run(run_id, "completed", result.stats)
            return result, paths
        except Exception:
            self.db.finish_run(run_id, "failed", result.stats)
            raise

    def _run(self, inp: ResearchInput, r: ResearchResult, website_id: int) -> dict[str, Path]:
        s = self.settings
        seed = inp.seed.strip()

        # [1] Crawl
        self.progress(1, "Crawling website")
        pages, reused, crawl_stats = self._crawl(website_id, inp.base_url, inp.fresh_crawl)
        r.pages, r.crawl_reused = pages, reused
        r.stats["crawl"] = crawl_stats
        if not pages:
            raise RuntimeError("No HTML pages could be crawled from the website (check robots.txt / availability).")

        # [2] Inventory
        self.progress(2, "Building content inventory")
        site_name = pages[0].title.split("|")[-1].strip() if pages[0].title else r.domain
        r.inventory = build_inventory(pages, site_name)
        self.db.save_page_topics(website_id, r.inventory)
        model = SemanticModel(s.use_sentence_transformers, s.embedding_model)
        corpus = [page_topic_text(p) for p in pages] + [i.primary_topic for i in r.inventory] + [seed]
        model.fit(corpus)
        r.semantic_backend = model.backend

        # [3] Cannibalization
        self.progress(3, "Detecting cannibalization")
        r.pairs, _ = detect_cannibalization(pages, r.inventory, model)
        self.db.executemany(
            "INSERT INTO cannibalization(run_id,website_id,url_a,url_b,risk,combined_score,data) VALUES(?,?,?,?,?,?,?)",
            [(r.run_id, website_id, p.url_a, p.url_b, p.risk.value, p.combined_score, p.model_dump_json()) for p in r.pairs],
        )

        # [4] SERP
        self.progress(4, "Researching Google SERPs & autocomplete")
        serp_provider = self._init_serp()
        r.serp = run_serp_research(serp_provider, self.db, r.run_id, seed, inp.country, inp.language,
                                   s.max_serp_queries, s.serp_results_per_query)
        if serp_provider is not None and r.serp.note:
            self.notes.append(r.serp.note)
        # Key-free: autocomplete expansion (Google/Bing/DuckDuckGo suggestions) + Wikipedia subtopics
        engines, wiki = self._init_free_sources(inp.language)
        free_stats = run_autocomplete_research(r.serp, engines, seed, inp.country, inp.language, s.autocomplete_depth)
        run_wikipedia_research(r.serp, wiki, parent_topic_for(seed), free_stats)
        r.stats["free_sources"] = {
            "autocomplete_probes": free_stats.probes, "autocomplete_suggestions": free_stats.suggestions,
            "engines_used": free_stats.engines_used, "engines_failed": free_stats.engines_failed,
            "wikipedia_article": free_stats.wikipedia_title, "wikipedia_sections": free_stats.wikipedia_sections,
            "requests": free_stats.calls,
        }
        if engines and not free_stats.engines_used:
            self.notes.append("Autocomplete endpoints returned nothing (offline, blocked or rate-limited) – see log.")
        elif free_stats.engines_failed:
            self.notes.append("Autocomplete engines without data: " + ", ".join(free_stats.engines_failed))

        # [5] PAA
        self.progress(5, "Researching PAA & question queries")
        expand_paa(serp_provider, r.serp, self.db, r.run_id, inp.country, inp.language)

        # [6] Community
        self.progress(6, "Researching Reddit/forums")
        reddit = self._init_reddit()
        r.community = run_community_research(seed, inp.country, inp.language, self.db, r.run_id, reddit,
                                             serp_provider if r.serp.available else None)
        if not r.community.available:
            self.notes.append("Community research produced no data: " + (r.community.note or "no sources configured"))

        # [7] Competitors
        self.progress(7, "Researching competitor content")
        r.competitors = run_competitor_research(r.serp, urlsplit(inp.base_url).netloc, r.inventory, model, self.db, r.run_id,
                                                s.crawl.user_agent, s.max_competitor_pages, s.crawl.timeout,
                                                manual_urls=s.competitor_urls)
        if r.competitors.note:
            self.notes.append(r.competitors.note)

        # [8] Generate (incl. AI-search research)
        self.progress(8, "Generating fan-out queries")
        llms = self._init_llms()
        r.llm = run_llm_research(llms, seed, inp.country, inp.language)
        if r.llm.note:
            self.notes.append(r.llm.note)
        seed_sims = model.similarity([seed], [i.primary_topic + " . " + i.title for i in r.inventory])[0] if r.inventory else np.zeros(0)
        relevant = seed_relevant_pages(seed, r.inventory, list(seed_sims))
        generator_llm = llms[0] if llms else None
        pool = generate_candidates(seed, inp.country, inp.language, r.serp, r.community, r.competitors, r.llm,
                                   r.inventory, relevant, generator_llm, target_pool=max(500, inp.num_queries * 5))
        candidates = list(pool.items.values())
        r.stats["candidates_generated"] = len(candidates)
        # refit TF-IDF vocabulary to include query language
        model.fit(corpus + [c.query for c in candidates])

        compute_topical_relevance(candidates, seed, model)
        candidates, dedup_stats = deduplicate(candidates, model, inp.country)
        r.stats["deduplication"] = dedup_stats

        arch = SiteArchitecture(r.inventory, inp.base_url)
        assess_coverage(candidates, r.inventory, pages, r.pairs, model, arch)
        kept, rejected = apply_quality_control(candidates, seed)
        r.stats["quality_control"] = {"kept": len(kept), "rejected": len(rejected)}

        # [9] Keyword metrics
        self.progress(9, "Fetching keyword metrics")
        planner, reason = self._init_ads()
        r.metrics = attach_keyword_metrics(kept, planner, inp.country or "", inp.language or "English", reason)
        r.stats["keyword_metrics"] = {"available": r.metrics.available, "with_volume": r.metrics.with_volume,
                                      "requested": r.metrics.requested}
        n_engines = len(free_stats.engines_used)
        for c in kept:
            c.demand_proxy = demand_proxy(r.serp.suggest_meta.get(c.normalized), n_engines)
        r.stats["keyword_metrics"]["with_estimated_demand_signal"] = sum(1 for c in kept if c.demand_proxy is not None)

        # [10] Score, cluster, select, recommend, export
        self.progress(10, "Selecting final opportunities")
        compute_serp_evidence(kept, r.serp, model)
        compute_business_relevance(kept, r.inventory, model)
        score_candidates(kept)
        parent = parent_topic_for(seed)
        cluster_queries(kept, seed, parent, model)
        final, sel_info = select_final(kept, inp.num_queries, model)
        r.stats["selection"] = sel_info
        r.internal_links = find_internal_links(final, r.inventory, pages, model)
        for c in kept:
            c.reasoning = _reasoning(c)
        r.final = final
        r.candidates = kept + rejected
        r.rejected = rejected
        r.topic_map = build_topic_map(kept, r.pairs)
        if self.skipped_apis:
            self.notes.insert(0, "Optional paid APIs not configured (skipped, nothing simulated): "
                              + ", ".join(dict.fromkeys(self.skipped_apis))
                              + ". search_volume stays empty; the score uses the ESTIMATED autocomplete demand signal instead.")
        r.data_notes = list(dict.fromkeys(self.notes))
        r.stats["content_gaps_identified"] = sum(1 for c in kept if c.coverage == Coverage.CONTENT_GAP)
        r.stats["api_calls"] = {
            **{f"free_{k}": v for k, v in free_stats.calls.items()},
            "serp": serp_provider.calls if serp_provider else 0,
            "reddit": reddit.calls if reddit else 0,
            "google_ads": planner.calls if planner else 0,
            **{f"llm_{p.name}": p.calls for p in llms},
        }
        self._persist(r)

        out = s.output_dir
        paths = {
            "xlsx": write_excel(r, out / "fanout_queries.xlsx"),
            "csv": write_csv(r, out / "fanout_queries.csv"),
        }
        paths.update(write_reports(r, out))
        log.info("Outputs written: %s", {k: str(v) for k, v in paths.items()})
        return paths

    # ------------------------------------------------------------ persistence
    def _persist(self, r: ResearchResult) -> None:
        final_ids = {id(c) for c in r.final}
        self.db.executemany(
            "INSERT OR REPLACE INTO keywords(run_id,keyword,role) VALUES(?,?,?)",
            [(r.run_id, r.seed, "seed")] + [(r.run_id, c.query, "final") for c in r.final],
        )
        self.db.executemany(
            "INSERT OR REPLACE INTO queries(run_id,query,normalized,selected,rejected_reason,score,data) VALUES(?,?,?,?,?,?,?)",
            [(r.run_id, c.query, c.normalized, int(id(c) in final_ids), c.rejected_reason, c.score,
              c.model_dump_json(exclude={"matching_pages"})) for c in r.candidates],
        )
        self.db.executemany(
            "INSERT INTO query_sources(run_id,query,source,evidence,label) VALUES(?,?,?,?,?)",
            [(r.run_id, c.query, o, " | ".join(c.evidence[:4]), c.label.value) for c in r.candidates for o in sorted(c.origins)],
        )
        self.db.executemany(
            "INSERT INTO recommendations(run_id,query,action,target_url,data) VALUES(?,?,?,?,?)",
            [(r.run_id, c.query, c.action.value if c.action else None, c.recommended_url,
              json.dumps({"content_type": c.content_type, "coverage": c.coverage.value if c.coverage else None,
                          "internal_links": c.internal_links, "content_gap": c.content_gap}))
             for c in r.final],
        )
        self.db.executemany(
            "INSERT INTO content_gaps(run_id,query,gap_type,data) VALUES(?,?,?,?)",
            [(r.run_id, c.query, c.coverage.value, json.dumps({"action": c.action.value if c.action else None,
                                                                "recommended_url": c.recommended_url, "note": c.content_gap}))
             for c in r.candidates if c.coverage in (Coverage.CONTENT_GAP, Coverage.PARTIALLY_COVERED) and not c.rejected_reason],
        )

    def close(self) -> None:
        self.db.close()
