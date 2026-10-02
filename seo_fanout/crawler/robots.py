"""robots.txt handling."""

from __future__ import annotations

import urllib.robotparser
from dataclasses import dataclass, field

from ..http import PoliteFetcher
from ..logging_setup import get_logger

log = get_logger("robots")


@dataclass
class RobotsInfo:
    available: bool
    text: str = ""
    crawl_delay: float | None = None
    sitemaps: list[str] = field(default_factory=list)
    parser: urllib.robotparser.RobotFileParser | None = None
    note: str = ""

    def allowed(self, url: str, user_agent: str) -> bool:
        if self.parser is None:
            return True
        try:
            return self.parser.can_fetch(user_agent, url)
        except Exception:
            return True


def load_robots(base_url: str, fetcher: PoliteFetcher, user_agent: str) -> RobotsInfo:
    robots_url = base_url.rstrip("/") + "/robots.txt"
    result = fetcher.fetch(robots_url, use_cache=False)
    if result.status is None:
        log.warning("robots.txt unavailable (%s) – crawling with conservative defaults", result.error)
        return RobotsInfo(available=False, note=f"robots.txt unreachable: {result.error}")
    if result.status in (401, 403):
        # RFC 9309: an unreachable-for-auth robots.txt => treat as full disallow.
        log.warning("robots.txt returned %s – treating site as disallowed", result.status)
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /"])
        return RobotsInfo(available=True, parser=parser, note=f"robots.txt HTTP {result.status}: disallow all")
    if result.status >= 400:
        log.info("No robots.txt (HTTP %s) – all paths allowed", result.status)
        return RobotsInfo(available=False, note=f"robots.txt HTTP {result.status}")

    parser = urllib.robotparser.RobotFileParser()
    lines = result.text.splitlines()
    parser.parse(lines)
    delay = None
    try:
        delay = parser.crawl_delay(user_agent) or parser.crawl_delay("*")
    except Exception:
        delay = None
    sitemaps = []
    for line in lines:
        if line.lower().startswith("sitemap:"):
            sitemaps.append(line.split(":", 1)[1].strip())
    log.info("robots.txt loaded: crawl-delay=%s sitemaps=%d", delay, len(sitemaps))
    return RobotsInfo(
        available=True, text=result.text[:50_000], crawl_delay=float(delay) if delay else None,
        sitemaps=sitemaps, parser=parser,
    )
