"""Competitor content analysis for pages ranking in the researched SERPs."""

from __future__ import annotations

import asyncio
import re
import urllib.robotparser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
import numpy as np

from ..analysis.embeddings import SemanticModel
from ..crawler.parser import parse_html
from ..crawler.urls import site_key
from ..db import Database
from ..logging_setup import get_logger
from ..models import CompetitorPage, PageInventory
from ..textutil import clean_text, is_question, key_phrases, normalize_query
from .serp_research import SerpResearch

log = get_logger("competitors")

EXCLUDED_DOMAINS = (
    "reddit.com", "quora.com", "youtube.com", "facebook.com", "instagram.com", "pinterest.com", "twitter.com",
    "x.com", "tiktok.com", "linkedin.com", "wikipedia.org", "amazon.", "flipkart.com", "ebay.", "google.",
)
GENERIC_HEADINGS = re.compile(
    r"^(related|recent|popular|share|comments?|leave a (reply|comment)|subscribe|newsletter|table of contents|"
    r"contents|follow us|about (us|the author)|you may also like|references|sources|search|menu|categories|tags|"
    r"conclusion|summary|final thoughts|faqs?|frequently asked questions|in this article|quick links|shop|cart)\b",
    re.I,
)


@dataclass
class CompetitorGap:
    topic: str
    gap_kind: str  # topic | question | comparison
    competitor_count: int
    competitor_urls: list[str]
    best_site_url: str | None
    site_similarity: float
    status: str  # NOT_COVERED | POORLY_COVERED | COVERED | COVERED_BY_MULTIPLE_PAGES
    site_urls_covering: list[str] = field(default_factory=list)


@dataclass
class CompetitorResearch:
    available: bool
    note: str = ""
    pages: list[CompetitorPage] = field(default_factory=list)
    gaps: list[CompetitorGap] = field(default_factory=list)


def select_competitor_urls(serp: SerpResearch, own_domain: str, limit: int) -> list[tuple[str, int]]:
    """Rank organic URLs by how often and how high they appear across SERPs."""
    score: dict[str, float] = defaultdict(float)
    best_pos: dict[str, int] = {}
    own = site_key(own_domain)
    for s in serp.serps:
        for r in s.organic:
            dom = r.domain
            netloc = site_key(urlsplit(r.url).netloc)
            if not dom or netloc == own or any(x in dom for x in EXCLUDED_DOMAINS):
                continue
            score[r.url] += 1.0 / (r.position + 1) + 0.15
            best_pos[r.url] = min(best_pos.get(r.url, 99), r.position)
    ranked = sorted(score, key=lambda u: -score[u])
    picked: list[tuple[str, int]] = []
    per_domain: Counter[str] = Counter()
    for url in ranked:
        dom = urlsplit(url).hostname or ""
        if per_domain[dom] >= 2:
            continue
        per_domain[dom] += 1
        picked.append((url, best_pos[url]))
        if len(picked) >= limit:
            break
    return picked


async def _fetch_one(client: httpx.AsyncClient, url: str, user_agent: str, sem: asyncio.Semaphore) -> tuple[str, int | None, str, dict]:
    async with sem:
        parts = urlsplit(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        try:
            r = await client.get(robots_url)
            if r.status_code < 400:
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(r.text.splitlines())
                if not rp.can_fetch(user_agent, url):
                    return url, None, "disallowed by robots.txt", {}
            await asyncio.sleep(1.0)  # polite gap between robots.txt and page on the same host
            resp = await client.get(url)
            if resp.status_code >= 400:
                return url, resp.status_code, f"HTTP {resp.status_code}", {}
            return url, resp.status_code, resp.text, {k.lower(): v for k, v in resp.headers.items()}
        except httpx.HTTPError as exc:
            return url, None, f"{type(exc).__name__}", {}


async def _fetch_all(urls: list[str], user_agent: str, timeout: float) -> list[tuple[str, int | None, str, dict]]:
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, trust_env=True, headers={"User-Agent": user_agent}
    ) as client:
        return await asyncio.gather(*[_fetch_one(client, u, user_agent, sem) for u in urls])


def _analyze(url: str, html: str, pos: int) -> CompetitorPage:
    host = urlsplit(url).hostname or ""
    page = parse_html(url, html, host)
    headings = [h for h in page.h2 + page.h3 if not GENERIC_HEADINGS.search(h)]
    stats = [
        clean_text(s) for s in re.split(r"(?<=[.!?])\s+", page.main_content)
        if re.search(r"\d+(\.\d+)?\s?%|\b\d{2,}\s?(people|users|patients|participants|studies|percent)\b|\bstudy\b", s, re.I)
        and 30 < len(s) < 260
    ][:15]
    comparisons = [h for h in headings if re.search(r"\bvs\.?\b|versus|compared|difference", h, re.I)]
    return CompetitorPage(
        url=url, domain=site_key(host), serp_position=pos, title=page.title, h1=page.h1,
        h2=[h for h in page.h2 if not GENERIC_HEADINGS.search(h)], h3=[h for h in page.h3 if not GENERIC_HEADINGS.search(h)],
        faq=page.faq_questions, entities=key_phrases(" . ".join(page.h1 + headings) + " . " + page.main_content[:15000], top_k=20),
        questions=list(dict.fromkeys([h for h in headings if is_question(h)] + page.faq_questions))[:40],
        comparisons=comparisons, statistics=stats, word_count=page.word_count, fetched=True,
    )


def _site_texts(inventory: list[PageInventory]) -> list[str]:
    return [
        " . ".join([i.primary_topic, i.title, i.h1] + i.secondary_topics[:15] + i.questions_answered[:15])
        for i in inventory
    ]


def compute_gaps(pages: list[CompetitorPage], inventory: list[PageInventory], model: SemanticModel,
                 covered: float = 0.55, poor: float = 0.38) -> list[CompetitorGap]:
    # Collect competitor topics with the URLs that cover them
    raw: list[tuple[str, str, str]] = []
    for p in pages:
        for h in p.h2 + p.h3:
            if 4 <= len(h) <= 140:
                raw.append((h, "comparison" if h in p.comparisons else ("question" if is_question(h) else "topic"), p.url))
        for q in p.faq:
            raw.append((q, "question", p.url))
    if not raw or not inventory:
        return []
    texts = [r[0] for r in raw]
    emb = model.encode(texts)
    # Greedy grouping of near-identical competitor topics
    groups: list[dict] = []
    for idx, (text, kind, url) in enumerate(raw):
        for g in groups:
            if float(emb[idx] @ emb[g["idx"]]) >= 0.8 or normalize_query(text) == normalize_query(g["text"]):
                g["urls"].add(url)
                break
        else:
            groups.append({"text": text, "kind": kind, "urls": {url}, "idx": idx})
    site_emb = model.encode(_site_texts(inventory))
    group_emb = emb[[g["idx"] for g in groups]]
    sim = SemanticModel.cosine_matrix(group_emb, site_emb)
    # Questions answered on the site are a strong coverage signal too
    site_questions = [(q, i.url) for i in inventory for q in i.questions_answered + i.secondary_topics]
    q_sim = None
    if site_questions:
        q_sim = SemanticModel.cosine_matrix(group_emb, model.encode([q for q, _ in site_questions]))

    gaps = []
    for gi, g in enumerate(groups):
        best = int(np.argmax(sim[gi])) if sim.shape[1] else -1
        best_sim = float(sim[gi, best]) if best >= 0 else 0.0
        best_url = inventory[best].url if best >= 0 else None
        if q_sim is not None and q_sim.shape[1]:
            qb = int(np.argmax(q_sim[gi]))
            if float(q_sim[gi, qb]) > best_sim:
                best_sim, best_url = float(q_sim[gi, qb]), site_questions[qb][1]
        covering = [inventory[j].url for j in np.where(sim[gi] >= covered)[0]]
        if len(covering) >= 2:
            status = "COVERED_BY_MULTIPLE_PAGES"
        elif best_sim >= covered:
            status = "COVERED"
        elif best_sim >= poor:
            status = "POORLY_COVERED"
        else:
            status = "NOT_COVERED"
        gaps.append(CompetitorGap(
            topic=g["text"], gap_kind=g["kind"], competitor_count=len(g["urls"]), competitor_urls=sorted(g["urls"]),
            best_site_url=best_url if best_sim >= poor else None, site_similarity=round(best_sim, 3),
            status=status, site_urls_covering=covering,
        ))
    gaps.sort(key=lambda g: (-g.competitor_count, g.status != "NOT_COVERED", g.topic))
    return gaps


def run_competitor_research(
    serp: SerpResearch, own_domain: str, inventory: list[PageInventory], model: SemanticModel,
    db: Database, run_id: str, user_agent: str, limit: int, timeout: float = 20.0,
    manual_urls: list[str] | None = None,
) -> CompetitorResearch:
    """Analyse competitor pages – URLs you supply (no API needed) plus top SERP results if available."""
    targets: list[tuple[str, int | None]] = []
    own = site_key(own_domain)
    for url in manual_urls or []:
        url = url.strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        if site_key(urlsplit(url).netloc) != own and url not in [t[0] for t in targets]:
            targets.append((url, None))
    if serp.available and serp.serps:
        for url, pos in select_competitor_urls(serp, own_domain, limit):
            if url not in [t[0] for t in targets]:
                targets.append((url, pos))
    targets = targets[: max(limit, len(manual_urls or []))]
    if not targets:
        return CompetitorResearch(available=False, note=(
            "No competitor pages analysed: pass competitor URLs with --competitors (no API needed) "
            "or configure a SERP provider."))
    pos_by_url = dict(targets)
    try:
        fetched = asyncio.run(_fetch_all([u for u, _ in targets], user_agent, timeout))
    except RuntimeError:  # already inside an event loop
        loop = asyncio.new_event_loop()
        fetched = loop.run_until_complete(_fetch_all([u for u, _ in targets], user_agent, timeout))
        loop.close()
    pages: list[CompetitorPage] = []
    for url, status, body, headers in fetched:
        if status is None or status >= 400 or "html" not in headers.get("content-type", "html"):
            log.info("Competitor page skipped: %s (%s)", url, body if status is None or status >= 400 else "non-HTML")
            pages.append(CompetitorPage(url=url, domain=site_key(urlsplit(url).hostname or ""), serp_position=pos_by_url[url],
                                        fetched=False, error=body if len(body) < 120 else f"HTTP {status}"))
            continue
        try:
            pages.append(_analyze(url, body, pos_by_url[url]))
        except Exception as exc:
            log.warning("Competitor parse failed for %s: %s", url, exc)
    ok_pages = [p for p in pages if p.fetched]
    gaps = compute_gaps(ok_pages, inventory, model)
    db.executemany(
        "INSERT INTO competitors(run_id,url,domain,serp_position,data) VALUES(?,?,?,?,?)",
        [(run_id, p.url, p.domain, p.serp_position, p.model_dump_json()) for p in pages],
    )
    log.info("Competitor research: %d/%d pages analysed, %d topic groups", len(ok_pages), len(pages), len(gaps))
    return CompetitorResearch(available=bool(ok_pages), pages=pages, gaps=gaps,
                              note="" if ok_pages else "Competitor pages could not be fetched (blocked or unavailable).")
