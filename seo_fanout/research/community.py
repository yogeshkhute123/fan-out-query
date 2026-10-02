"""Reddit / Quora / forum research.

Community content is used to learn *how users phrase needs* and which concerns
recur – never as a source of facts. Confidence grows with the number of
independent threads and platforms that raise the same question.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from ..db import Database
from ..logging_setup import get_logger
from ..models import CommunityItem, CommunityQuestion
from ..providers.locale import country_info, language_info
from ..providers.reddit import RedditProvider
from ..providers.serp import BaseSerpProvider
from ..textutil import clean_text, extract_question_sentences, is_question, key_phrases, normalize_query, stem_tokens

log = get_logger("community")

FORUM_HINT = re.compile(r"forum|community|discuss|board|talk|answers|stackexchange|/threads?/|/t/|/topic/|/questions/", re.I)

CATEGORY_RULES = [
    ("comparison", re.compile(r"\bvs\.?\b|versus|better than|\bor\b.*\?|which (one|is better)|compared", re.I)),
    ("objection", re.compile(r"worth it|overrated|scam|waste|too expensive|not worth|hype|any good|legit", re.I)),
    ("misconception", re.compile(r"\bmyth|is it true|really need|actually|do i really|supposed to|safe to", re.I)),
    ("buying_concern", re.compile(r"\bbuy|price|cost|fake|original|where (can|do|to)|brand|recommend|suggest|budget|cheap", re.I)),
    ("problem", re.compile(r"problem|issue|help|not working|doesn'?t work|stopped|broke|irritat|rash|greasy|oily|pimple|"
                           r"breakout|burn|fix|wrong|stuck|struggl", re.I)),
]


@dataclass
class CommunityResearch:
    available: bool
    note: str = ""
    items: list[CommunityItem] = field(default_factory=list)
    questions: list[CommunityQuestion] = field(default_factory=list)
    terminology: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)


def _platform(url: str) -> tuple[str, str]:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    if "reddit.com" in host:
        return "reddit", host
    if "quora.com" in host:
        return "quora", host
    if FORUM_HINT.search(url):
        return "forum", host
    return "web", host


def _categorize(question: str) -> str:
    for name, pattern in CATEGORY_RULES:
        if pattern.search(question):
            return name
    return "question"


def _candidate_questions(item: CommunityItem, comments: list[str]) -> list[str]:
    out: list[str] = []
    title = clean_text(re.sub(r"\s*[-|:]\s*(Quora|Reddit|r/\w+).*$", "", item.title, flags=re.I))
    title = re.sub(r"^r/\w+\s*[-:]\s*", "", title)
    if is_question(title) and 10 <= len(title) <= 180:
        out.append(title.rstrip("?") + "?")
    elif 10 <= len(title) <= 140:
        out.append(title)  # non-question titles still show problems/terminology
    out += extract_question_sentences(item.text, limit=5)
    for c in comments[:15]:
        out += extract_question_sentences(c, limit=2)
    return out


def _merge(questions: list[tuple[str, str, str]], seed_stems: set[str]) -> list[CommunityQuestion]:
    """Group (question, platform, url) tuples into de-duplicated questions."""
    groups: list[dict] = []
    for text, platform, url in questions:
        norm = normalize_query(text)
        stems = set(stem_tokens(text))
        if not norm or not (stems & seed_stems):
            continue  # unrelated to the topic
        placed = False
        for g in groups:
            if g["norm"] == norm or SequenceMatcher(None, g["norm"], norm).ratio() >= 0.86:
                g["urls"].add(url)
                g["platforms"].add(platform)
                placed = True
                break
        if not placed:
            groups.append({"text": text, "norm": norm, "urls": {url}, "platforms": {platform}})
    result = []
    for g in groups:
        freq = len(g["urls"])
        independent_domains = len({urlsplit(u).hostname for u in g["urls"]})
        if freq >= 3 and independent_domains >= 2:
            conf = "HIGH"
        elif freq >= 2:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        result.append(
            CommunityQuestion(
                question=g["text"], sources=sorted(g["platforms"]), urls=sorted(g["urls"])[:10],
                frequency=freq, category=_categorize(g["text"]), confidence=conf,
            )
        )
    result.sort(key=lambda q: (-q.frequency, q.question))
    return result


def run_community_research(
    seed: str, country: str, language: str, db: Database, run_id: str,
    reddit: RedditProvider | None, serp: BaseSerpProvider | None, max_threads_with_comments: int = 6,
) -> CommunityResearch:
    research = CommunityResearch(available=False)
    notes = []
    collected: list[tuple[str, str, str]] = []
    texts: list[str] = []

    # 1. Reddit official API
    if reddit is not None:
        try:
            items: list[CommunityItem] = []
            for q in (seed, f"{seed} advice", f"{seed} recommendation"):
                items += reddit.search(q, limit=25)
            seen = set()
            unique = [i for i in items if not (i.thread_url in seen or seen.add(i.thread_url))]
            unique.sort(key=lambda i: -(i.num_comments or 0))
            for idx, item in enumerate(unique):
                comments = reddit.top_comments(item.thread_url) if idx < max_threads_with_comments else []
                for q in _candidate_questions(item, comments):
                    collected.append((q, "reddit", item.thread_url))
                texts.append(item.title + " " + item.text + " " + " ".join(comments))
            research.items += unique
            research.sources_used.append("reddit_api")
        except Exception as exc:
            notes.append(f"Reddit API failed: {exc}")
            log.warning("Reddit research failed: %s", exc)
    else:
        notes.append("Reddit API not configured")

    # 2. Quora / Reddit / forums discovered through the SERP API (site: operators)
    if serp is not None:
        gl, _, google_domain = country_info(country)
        hl, _ = language_info(language)
        for q in (f"site:quora.com {seed}", f"site:reddit.com {seed}", f"{seed} forum", f"{seed} community discussion"):
            data = serp.search(q, gl, hl, num=10, google_domain=google_domain)
            if data.error:
                continue
            for r in data.organic:
                platform, host = _platform(r.url)
                if platform == "web":
                    continue
                item = CommunityItem(source=platform, platform_domain=host, thread_url=r.url, title=r.title, text=r.snippet)
                research.items.append(item)
                for cq in _candidate_questions(item, []):
                    collected.append((cq, platform, r.url))
                texts.append(r.title + " " + r.snippet)
            for paa in data.paa:
                collected.append((paa["question"], "forum" if "forum" in q else "quora" if "quora" in q else "reddit",
                                  paa.get("link") or f"paa:{q}"))
        research.sources_used.append(f"serp:{serp.name}")
    else:
        notes.append("SERP provider not configured (Quora/forum discovery skipped)")

    research.available = bool(research.items)
    research.questions = _merge(collected, set(stem_tokens(seed)))
    research.terminology = key_phrases(" . ".join(texts), top_k=30) if texts else []
    research.note = "; ".join(notes)
    db.executemany(
        "INSERT INTO community_questions(run_id,question,category,frequency,sources,urls,confidence) VALUES(?,?,?,?,?,?,?)",
        [(run_id, q.question, q.category, q.frequency, ",".join(q.sources), "\n".join(q.urls), q.confidence)
         for q in research.questions],
    )
    log.info("Community research: %d items, %d distinct questions", len(research.items), len(research.questions))
    return research
