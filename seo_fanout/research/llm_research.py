"""AI-search behaviour research through official LLM APIs (never web-UI scraping).

Everything returned here is labelled LLM_GENERATED: it shows how AI assistants
frame the topic (follow-ups, comparisons, entities), not real search demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..logging_setup import get_logger
from ..models import LLMInsight
from ..providers.llm import LLMProvider

log = get_logger("llm_research")

CATEGORIES = (
    "conversational_questions",
    "follow_up_questions",
    "comparison_questions",
    "recommendation_questions",
    "decision_questions",
    "problem_solving_queries",
    "alternative_queries",
)

SYSTEM = (
    "You are a search-behaviour researcher. You describe how people phrase questions to AI assistants and "
    "search engines about a topic. Never invent statistics, search volumes, prices, brand claims or study results. "
    "Reply with a single JSON object only."
)

PROMPT = """Topic (seed keyword): "{seed}"
Market: {country}; language: {language}.

List the questions and follow-ups real users typically ask AI assistants (ChatGPT, Perplexity, Gemini, Google AI Mode)
when researching this topic. Each item must represent a DISTINCT information need – no rephrasings of the same question.

Return JSON with exactly these keys (arrays of strings, 6-12 items each, written in {language}):
{{
  "conversational_questions": [...],
  "follow_up_questions": [...],
  "comparison_questions": [...],
  "recommendation_questions": [...],
  "decision_questions": [...],
  "problem_solving_queries": [...],
  "alternative_queries": [...],
  "entities": [{{"entity": "...", "relationship": "how it relates to the topic"}}]
}}"""


@dataclass
class LLMResearch:
    available: bool
    note: str = ""
    insights: list[LLMInsight] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    providers_failed: list[str] = field(default_factory=list)


def run_llm_research(providers: list[LLMProvider], seed: str, country: str, language: str) -> LLMResearch:
    research = LLMResearch(available=False)
    if not providers:
        research.note = "No LLM provider configured – AI-search research skipped (nothing is simulated)."
        return research
    prompt = PROMPT.format(seed=seed, country=country or "global", language=language or "English")
    for provider in providers:
        data, resp = provider.complete_json(SYSTEM, prompt, max_tokens=3000)
        if not isinstance(data, dict) or resp is None:
            research.providers_failed.append(provider.name)
            continue
        research.providers_used.append(provider.name)
        for cat in CATEGORIES:
            for text in data.get(cat, []) or []:
                if isinstance(text, str) and 5 < len(text) < 200:
                    research.insights.append(
                        LLMInsight(provider=provider.name, model=provider.model, category=cat, text=text.strip(),
                                   citations=resp.citations[:10])
                    )
        for ent in data.get("entities", []) or []:
            if isinstance(ent, dict) and ent.get("entity"):
                research.entities.append({"provider": provider.name, "entity": str(ent["entity"]),
                                          "relationship": str(ent.get("relationship", ""))})
    research.available = bool(research.insights)
    if research.providers_failed:
        research.note = "Failed providers: " + ", ".join(research.providers_failed)
    log.info("LLM research: %d insights from %s", len(research.insights), ", ".join(research.providers_used) or "none")
    return research
