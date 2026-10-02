"""Responsible website crawler (robots.txt + sitemaps + internal links)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ..config import CrawlSettings
from ..db import Database
from ..http import PoliteFetcher
from ..logging_setup import get_logger
from ..models import PageData
from .parser import parse_html
from .robots import RobotsInfo, load_robots
from .sitemap import discover_sitemap_urls
from .urls import dedup_key, is_crawlable_path, is_pagination, is_same_site, normalize_url, url_depth

log = get_logger("crawler")


@dataclass
class CrawlStats:
    discovered: int = 0
    fetched: int = 0
    from_cache: int = 0
    html_pages: int = 0
    indexable: int = 0
    skipped_robots: int = 0
    skipped_non_html: int = 0
    duplicates: int = 0
    failed: int = 0
    status_counts: dict[int, int] = field(default_factory=dict)
    sitemap_urls: int = 0
    sitemaps: list[str] = field(default_factory=list)
    robots_note: str = ""


class WebsiteUnavailableError(RuntimeError):
    pass


class Crawler:
    def __init__(self, base_url: str, settings: CrawlSettings, db: Database, website_id: int):
        self.base_url = base_url.rstrip("/")
        self.base_host = urlsplit(base_url).hostname or ""
        self.settings = settings
        self.db = db
        self.website_id = website_id
        self.fetcher = PoliteFetcher(
            user_agent=settings.user_agent,
            delay=settings.crawl_delay,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            db=db,
        )
        self.stats = CrawlStats()
        self.robots: RobotsInfo | None = None

    def close(self) -> None:
        self.fetcher.close()

    # ------------------------------------------------------------------
    def _priority(self, url: str, from_sitemap: bool) -> tuple[int, int, str]:
        # Lower = earlier. Sitemap URLs and shallow pages first; pagination last.
        penalty = 5 if is_pagination(url) else 0
        return (penalty + (0 if from_sitemap else 1), url_depth(url), url)

    def crawl(self) -> list[PageData]:
        s = self.settings
        # 1. homepage reachability
        home = self.fetcher.fetch(self.base_url + "/", use_cache=False)
        if home.status is None:
            raise WebsiteUnavailableError(f"Website unreachable: {home.error}")
        if home.status >= 500:
            raise WebsiteUnavailableError(f"Website returned HTTP {home.status}")
        # Respect a redirect of the homepage to another host (e.g. non-www -> www)
        final_host = urlsplit(home.final_url).hostname or self.base_host
        if final_host != self.base_host and is_same_site(home.final_url, self.base_host):
            log.info("Homepage redirected to %s – using it as the crawl host", final_host)
            self.base_host = final_host
            self.base_url = f"{urlsplit(home.final_url).scheme}://{urlsplit(home.final_url).netloc}"

        # 2. robots.txt
        if s.respect_robots:
            self.robots = load_robots(self.base_url, self.fetcher, s.user_agent)
            self.stats.robots_note = self.robots.note
            if self.robots.crawl_delay:
                delay = min(self.robots.crawl_delay, 30.0)
                self.fetcher.set_host_delay(urlsplit(self.base_url).netloc, delay)
                log.info("Honouring robots.txt crawl-delay of %.1fs", delay)
        else:
            self.robots = RobotsInfo(available=False, note="robots.txt checks disabled by user")

        # 3. sitemaps
        sitemap_urls, sitemaps = discover_sitemap_urls(
            self.base_url, self.fetcher, self.robots.sitemaps if self.robots else [],
            max_urls=s.max_sitemap_urls, max_sitemaps=s.max_sitemaps,
        )
        self.stats.sitemaps = sitemaps
        self.stats.sitemap_urls = len(sitemap_urls)

        # 4. frontier
        frontier: list[tuple[tuple[int, int, str], str, int]] = []
        seen: set[str] = set()

        def enqueue(raw: str, from_sitemap: bool, depth: int) -> None:
            url = normalize_url(raw)
            if not url or not is_same_site(url, self.base_host):
                return
            key = dedup_key(url)
            if key in seen:
                self.stats.duplicates += 1
                return
            if not is_crawlable_path(url):
                self.stats.skipped_non_html += 1
                return
            seen.add(key)
            self.stats.discovered += 1
            heapq.heappush(frontier, (self._priority(url, from_sitemap), url, depth))

        enqueue(self.base_url + "/", True, 0)
        for u in sitemap_urls:
            enqueue(u, True, 0)

        pages: list[PageData] = []
        content_hashes: dict[str, str] = {}
        while frontier and len(pages) < s.max_pages:
            _, url, depth = heapq.heappop(frontier)
            if self.robots and not self.robots.allowed(url, s.user_agent):
                self.stats.skipped_robots += 1
                log.debug("Disallowed by robots.txt: %s", url)
                continue
            result = self.fetcher.fetch(url)
            self.stats.fetched += 1
            if result.from_cache:
                self.stats.from_cache += 1
            if result.status is None:
                self.stats.failed += 1
                log.warning("Failed page: %s (%s)", url, result.error)
                continue
            self.stats.status_counts[result.status] = self.stats.status_counts.get(result.status, 0) + 1
            if result.status in (403, 401):
                log.warning("HTTP %s (access denied) for %s", result.status, url)
                continue
            if result.status >= 400:
                log.info("Skipping %s – HTTP %s", url, result.status)
                continue
            if not result.is_html:
                self.stats.skipped_non_html += 1
                continue
            # Redirected outside the site?
            final = normalize_url(result.final_url) or url
            if not is_same_site(final, self.base_host):
                log.info("Skipping %s – redirects off-site to %s", url, final)
                continue
            if final != url and dedup_key(final) in seen and dedup_key(final) != dedup_key(url):
                self.stats.duplicates += 1
                continue
            if not result.text.strip():
                log.info("Empty page skipped: %s", url)
                continue
            try:
                page = parse_html(final, result.text, self.base_host, result.status, result.headers)
            except Exception as exc:  # malformed HTML should never stop the crawl
                self.stats.failed += 1
                log.warning("Could not parse %s: %s", url, exc)
                continue
            page.final_url = final
            page.depth = depth
            if page.indexable and page.content_hash in content_hashes and page.word_count > 50:
                page.indexable = False
                page.index_reason = f"duplicate content of {content_hashes[page.content_hash]}"
                self.stats.duplicates += 1
            elif page.indexable:
                content_hashes[page.content_hash] = final
            pages.append(page)
            self.db.save_page(self.website_id, page)
            self.stats.html_pages += 1
            if page.indexable:
                self.stats.indexable += 1
            if len(pages) % 25 == 0:
                log.info("Crawl progress: %d pages (%d in frontier)", len(pages), len(frontier))

            # canonical target is a strong discovery signal
            if page.canonical and page.canonical != final:
                enqueue(page.canonical, True, depth)
            if s.follow_internal_links and depth < s.max_depth and "nofollow" not in page.robots_meta.lower():
                for link in page.internal_links:
                    enqueue(link, False, depth + 1)

        self.db.mark_crawled(
            self.website_id, len(pages), self.robots.text if self.robots else None, self.stats.sitemaps
        )
        log.info(
            "Crawl finished: %d pages (%d indexable), %d robots-skipped, %d duplicates, %d failed",
            len(pages), self.stats.indexable, self.stats.skipped_robots, self.stats.duplicates, self.stats.failed,
        )
        return pages
