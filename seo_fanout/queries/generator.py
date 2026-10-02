"""Fan-out candidate generation.

Candidates come from (in order of evidential weight):
  1. real observations – PAA, related searches, autocomplete, community questions,
     competitor headings;
  2. LLM generation grounded in that evidence (labelled LLM_GENERATED);
  3. rule-based templates across the fan-out dimensions (labelled TEMPLATE_GENERATED),
     with audience/situational modifiers only when the evidence shows they matter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..logging_setup import get_logger
from ..models import DataLabel, PageInventory, QueryCandidate
from ..providers.llm import LLMProvider
from ..research.community import CommunityResearch
from ..research.competitors import CompetitorResearch
from ..research.llm_research import LLMResearch
from ..research.serp_research import SerpResearch
from ..textutil import STOPWORDS, clean_text, content_tokens, normalize_query, singularize, stem_tokens

log = get_logger("generator")

REAL_ORIGINS = {"google", "paa", "autocomplete", "related", "reddit", "quora", "forums", "competitors"}


@dataclass
class SeedParts:
    seed: str
    head: str
    qualifier: str | None
    connector: str | None


def split_seed(seed: str) -> SeedParts:
    s = normalize_query(seed)
    m = re.match(r"^(.+?)\s+(for|in|with|on|near|without)\s+(.+)$", s)
    if m:
        return SeedParts(seed=s, head=m.group(1), qualifier=m.group(3), connector=m.group(2))
    return SeedParts(seed=s, head=s, qualifier=None, connector=None)


AUDIENCE_TERMS = {
    "beginners": r"beginner|first time|newbie|starter",
    "men": r"\bmen\b|\bmale\b|\bguys?\b|\bman\b",
    "women": r"\bwomen\b|\bfemale\b|\bladies\b|\bwoman\b",
    "kids": r"\bkids?\b|\bchildren\b|\bchild\b|\bbab(y|ies)\b|toddler",
    "teenagers": r"\bteens?\b|teenager",
    "seniors": r"senior|elderly|older adults|over 50|over 60|mature",
    "professionals": r"professional|business|enterprise|agency|freelancer",
    "students": r"student|college",
}
# situation -> (evidence pattern, natural phrase appended to the seed)
SITUATION_TERMS = {
    "travel": (r"travel|trip|vacation|holiday", "while travelling"),
    "summer": (r"summer|hot weather|humid", "in summer"),
    "winter": (r"winter|cold weather", "in winter"),
    "office": (r"office|work(ing)? from|desk|indoor", "for indoor and office use"),
    "workout": (r"workout|gym|sport|running|exercise|sweat", "during workouts"),
    "monsoon": (r"monsoon|rainy", "in monsoon"),
    "daily use": (r"daily|everyday|every day", "for daily use"),
    "outdoor": (r"outdoor|beach|hiking|swim", "for outdoor activities"),
}
VERBISH = {
    "buy", "shop", "apply", "reapply", "use", "using", "wear", "wearing", "get", "make", "choose", "find", "order",
    "add", "best", "top", "how", "why", "what", "our", "your", "new", "sale", "view", "read", "click", "free",
    "product", "products", "detail", "details", "complete", "ultimate", "different", "every", "each", "any", "other",
    "same", "good", "great", "right", "perfect", "favourite", "favorite", "popular", "much", "more", "many", "own",
    "need", "needs", "skip", "skipping", "putting", "doesn't", "does", "which", "this", "that", "the", "a", "an",
    "little", "enough", "regular", "normal", "usual", "first", "last", "only", "real", "actual",
}


def _templates(p: SeedParts, country: str, evidence_text: str, has_locations: bool) -> list[tuple[str, str]]:
    s, h, q = p.seed, p.head, p.qualifier
    t: list[tuple[str, str]] = []
    add = lambda dim, *qs: t.extend((dim, x) for x in qs)  # noqa: E731

    # Core information
    if q:
        add("core", f"what makes a {h} good for {q}", f"why use {s}", f"is {h} necessary for {q}",
            f"when to use {s}", f"who needs {s}", f"how does {h} work for {q}")
    else:
        add("core", f"what is {s}", f"why use {s}", f"how does {s} work", f"when to use {s}",
            f"who needs {s}", f"where to use {s}")
    # How-to
    add("howto", f"how to use {s}", f"how to choose {s}", f"how to compare {h} options" + (f" for {q}" if q else ""),
        f"how to get the best results from {s}", f"how much {h} to use" + (f" for {q}" if q else ""),
        f"how often to use {s}")
    # Problems
    add("problems", f"common mistakes with {s}", f"why is {s} not working", f"{s} problems and how to fix them",
        f"{s} side effects")
    if q:
        add("problems", f"can {h} make {q} worse", f"what to avoid in {s}")
    # Benefits
    add("benefits", f"benefits of {s}", f"is {s} worth it", f"what results to expect from {s}")
    # Comparisons
    add("comparison", f"best alternatives to {s}")
    if q:
        add("comparison", f"which type of {h} is best for {q}", f"difference between {h} types for {q}")
    # Commercial
    add("commercial", f"best {s}", f"{s} reviews", f"{s} price", f"buy {s} online", f"top {h} brands" + (f" for {q}" if q else ""),
        f"affordable {s}")
    # Local
    if country:
        add("local", f"best {s} in {country.lower()}")
    if has_locations or re.search(r"near me", evidence_text):
        add("local", f"{s} near me")
    # Conversational / AI-search
    add("conversational",
        f"what should i know before choosing {s}",
        f"how do i choose the right {h}" + (f" for {q}" if q else ""),
        f"what are the common mistakes people make with {s}",
        f"what should i compare before buying {s}")
    # Audience / situational – only when evidence mentions them
    add("audience", f"beginner's guide to {s}")
    for name, pattern in AUDIENCE_TERMS.items():
        if name != "beginners" and re.search(pattern, evidence_text, re.I):
            add("audience", f"{s} for {name}")
    for name, (pattern, phrase) in SITUATION_TERMS.items():
        if re.search(pattern, evidence_text, re.I):
            add("situational", f"{s} {phrase}")
    return t


def observed_variants(p: SeedParts, texts: list[tuple[str, str]], min_count: int = 2, limit: int = 8) -> list[str]:
    """'<modifier> <head>' noun phrases observed in real text, e.g. 'gel sunscreen', 'mineral sunscreen'.

    ``texts`` are (source_id, text) pairs; a variant must appear in at least ``min_count``
    independent sources (pages, competitor URLs, SERP features).
    """
    head_words = normalize_query(p.head).split()
    if not head_words:
        return []
    head_last = singularize(head_words[-1])
    seed_stems = set(stem_tokens(p.seed))
    sources: dict[str, set[str]] = {}
    for source_id, text in texts:
        words = normalize_query(text).split()
        for a, b in zip(words, words[1:]):
            if singularize(b) != head_last:
                continue
            if a in STOPWORDS or a in VERBISH or a.isdigit() or len(a) < 3 or singularize(a) in seed_stems:
                continue
            sources.setdefault(a, set()).add(source_id)
    counts = {m: len(s) for m, s in sources.items()}
    ranked = sorted((m for m, n in counts.items() if n >= min_count), key=lambda m: -counts[m])
    return [f"{m} {head_last}" for m in ranked[:limit]]


def _entity_templates(p: SeedParts, variants: list[str], limit: int = 12) -> list[tuple[str, str]]:
    """Templates that combine the seed with vocabulary actually observed on relevant pages."""
    out: list[tuple[str, str]] = []
    for v in variants[: limit // 2]:
        out.append(("entity", f"is {v} good for {p.qualifier}" if p.qualifier else f"what is {v}"))
    for a, b in zip(variants[: limit // 3], variants[1 : limit // 3 + 1]):
        out.append(("comparison", f"{a} vs {b}" + (f" for {p.qualifier}" if p.qualifier else "")))
    return out


LLM_SYSTEM = (
    "You are an expert SEO and GEO researcher generating fan-out queries. Each query must express a DISTINCT "
    "search need (not a reword of another). Never output search volumes or statistics. Output one JSON object only."
)

LLM_PROMPT = """Seed keyword: "{seed}"
Market: {country}; language: {language}
Website context (pages that already exist on the target site): {site}

Real evidence collected from Google SERPs, People Also Ask, Reddit/Quora/forums and competitor pages:
{evidence}

Generate up to {n} fan-out search queries for this seed, spread across these dimensions:
core (what/why/how/when/where/who), howto (use, choose, compare, improve, fix), problems (mistakes, troubleshooting,
side effects where relevant), benefits (advantages, use cases, results), comparison (A vs B, alternatives,
differences, which to choose), audience (only if genuinely relevant), situational (only if genuinely relevant),
commercial (best, reviews, price, buy, brands), local (only if relevant), conversational (natural AI-assistant
questions).

Rules: write queries the way real users type or ask them; prefer specific long-tail needs; no keyword permutations;
no brand names unless they appear in the evidence; lower-case.
Return: {{"queries": [{{"query": "...", "dimension": "..."}}]}}"""


class CandidatePool:
    def __init__(self) -> None:
        self.items: dict[str, QueryCandidate] = {}

    def add(self, query: str, origin: str, evidence: str = "", dimension: str = "", label: DataLabel | None = None) -> None:
        query = clean_text(query).strip(" ?.!").replace("’", "'")
        norm = normalize_query(query)
        if not norm or len(norm) < 4:
            return
        label = label or (DataLabel.REAL if origin in REAL_ORIGINS else DataLabel.LLM if origin == "llm" else DataLabel.TEMPLATE)
        cand = self.items.get(norm)
        if cand is None:
            cand = QueryCandidate(query=norm, normalized=norm, dimension=dimension, label=label)
            self.items[norm] = cand
        cand.origins.add(origin)
        if evidence and evidence not in cand.evidence and len(cand.evidence) < 8:
            cand.evidence.append(evidence[:200])
        if not cand.dimension and dimension:
            cand.dimension = dimension
        # Real evidence trumps generated provenance
        rank = {DataLabel.REAL: 0, DataLabel.LLM: 1, DataLabel.TEMPLATE: 2}
        if rank.get(label, 3) < rank.get(cand.label, 3):
            cand.label = label

    def __len__(self) -> int:
        return len(self.items)


def _query_like(text: str, max_words: int = 16) -> bool:
    words = normalize_query(text).split()
    return 2 <= len(words) <= max_words


def generate_candidates(
    seed: str, country: str, language: str,
    serp: SerpResearch, community: CommunityResearch, competitors: CompetitorResearch, llm_research: LLMResearch,
    inventory: list[PageInventory], relevant_pages: list[PageInventory],
    llm_generator: LLMProvider | None, target_pool: int = 500,
) -> CandidatePool:
    pool = CandidatePool()
    parts = split_seed(seed)

    # 1. Google SERP evidence (real)
    for entry in serp.paa.values():
        pool.add(entry["question"], "paa", f"PAA for: {', '.join(entry['source_queries'][:2])}", "paa")
    for rel, count in serp.related.items():
        pool.add(rel, "related", f"Related search (seen {count}x)", "serp")
    for base, suggestions in serp.autocomplete.items():
        for sug in suggestions:
            pool.add(sug, "autocomplete", f"Autocomplete for '{base}'", "serp")
    # Only the seed itself is taken from the researched patterns: the patterns ("how to <seed>",
    # "why <seed>") are probes for SERP features, not necessarily queries people type.
    for s in serp.serps:
        if s.organic and normalize_query(s.query) == normalize_query(seed):
            pool.add(s.query, "google", f"SERP researched: {len(s.organic)} organic results", "serp")

    # 2. Community (real)
    for cq in community.questions:
        if _query_like(cq.question, 18):
            for src in cq.sources:
                origin = {"reddit": "reddit", "quora": "quora"}.get(src, "forums")
                pool.add(cq.question, origin, f"{src} ({cq.frequency} thread(s)): {cq.urls[0] if cq.urls else ''}", "community")

    # 3. Competitor headings (real observations of covered needs)
    seed_stems = set(stem_tokens(seed))
    for gap in competitors.gaps:
        topic = gap.topic
        if not _query_like(topic, 14):
            continue
        if gap.gap_kind == "question" or set(stem_tokens(topic)) & seed_stems:
            pool.add(topic, "competitors", f"Covered by {gap.competitor_count} competitor(s): {gap.competitor_urls[0]}", "competitor")

    # 4. LLM AI-search research (LLM-generated)
    for ins in llm_research.insights:
        pool.add(ins.text, "llm", f"{ins.provider}: {ins.category}", "conversational" if "question" in ins.category else ins.category)

    real_count = len(pool)

    # 5. LLM fan-out generation grounded in the evidence
    if llm_generator is not None:
        evidence_lines = (
            [f"PAA: {e['question']}" for e in list(serp.paa.values())[:40]]
            + [f"Related: {r}" for r in list(serp.related)[:30]]
            + [f"Community: {q.question}" for q in community.questions[:40]]
            + [f"Competitor heading: {g.topic}" for g in competitors.gaps[:40]]
        )
        site_desc = "; ".join(f"{p.primary_topic} ({p.page_type.value})" for p in relevant_pages[:25]) or "no closely related pages"
        n = max(80, min(300, target_pool - len(pool)))
        data, resp = llm_generator.complete_json(
            LLM_SYSTEM,
            LLM_PROMPT.format(seed=seed, country=country or "global", language=language or "English", site=site_desc,
                              evidence="\n".join(evidence_lines) or "(no external evidence available)", n=n),
            max_tokens=12000,
        )
        if isinstance(data, dict):
            for item in data.get("queries", []) or []:
                if isinstance(item, dict) and isinstance(item.get("query"), str):
                    pool.add(item["query"], "llm", f"{llm_generator.name} fan-out generation", str(item.get("dimension", "")))
                elif isinstance(item, str):
                    pool.add(item, "llm", f"{llm_generator.name} fan-out generation", "")
        log.info("LLM fan-out generation added %d candidates", len(pool) - real_count)

    # 6. Template generation (always – transparent, labelled)
    evidence_text = " ".join(
        list(serp.related) + [e["question"] for e in serp.paa.values()] + [q.question for q in community.questions]
        + community.terminology + [g.topic for g in competitors.gaps]
    ).lower()
    has_locations = any(p.page_type.value == "Location page" for p in inventory)
    for dim, q in _templates(parts, country, evidence_text, has_locations):
        pool.add(q, "template", f"pattern: {dim}", dim)
    observed_text: list[tuple[str, str]] = []
    for p in relevant_pages[:15]:
        observed_text += [(p.url, t) for t in [p.title, p.h1] + p.secondary_topics + p.questions_answered]
    for p in competitors.pages:
        observed_text += [(p.url, t) for t in [p.title] + p.h1 + p.h2 + p.h3 + p.faq]
    observed_text += [(f"related:{r}", r) for r in serp.related]
    observed_text += [(f"paa:{e['question']}", e["question"]) for e in serp.paa.values()]
    observed_text += [(u, q.question) for q in community.questions for u in q.urls[:3]]
    for dim, q in _entity_templates(parts, observed_variants(parts, observed_text)):
        pool.add(q, "template", "pattern: observed entity", dim)

    log.info("Candidate pool: %d (real-evidence %d)", len(pool), real_count)
    return pool


def seed_relevant_pages(seed: str, inventory: list[PageInventory], sims: list[float], k: int = 25) -> list[PageInventory]:
    ranked = sorted(zip(inventory, sims), key=lambda x: -x[1])
    return [p for p, s in ranked[:k] if s > 0.15 or set(content_tokens(seed)) & set(content_tokens(p.primary_topic))]
