"""Rule-based search-intent classification.

Deterministic and explainable: each classification returns the matched signal
so the reasoning column can show *why* a query received its intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Intent
from ..textutil import normalize_query

_P = lambda s: re.compile(s, re.I)  # noqa: E731

RULES: list[tuple[Intent, re.Pattern, float, str]] = [
    (Intent.LOCAL, _P(r"\bnear me\b|\bnearby\b|\bin my area\b|\bclosest\b|\bopen now\b|\bstore(s)? in\b|\bshops? in\b"), 0.9, "local modifier"),
    (Intent.TRANSACTIONAL, _P(r"\b(buy|order|purchase|shop|coupon|discount|deal|deals|promo code|free shipping|subscribe|book|hire|for sale|add to cart|online store)\b"), 0.9, "purchase verb"),
    (Intent.COMPARISON, _P(r"\bvs\.?\b|\bversus\b|\bcompared? (to|with)\b|\bcomparison\b|\bdifference between\b|\bor\b.+\bwhich\b|\bbetter than\b|\balternatives?\b|\binstead of\b"), 0.85, "comparison phrasing"),
    (Intent.TRANSACTIONAL_INVESTIGATION, _P(r"\b(price|prices|pricing|cost|costs|cheap|cheapest|affordable|budget|under \d+|worth (it|the money)|where to buy)\b"), 0.8, "price / purchase-research modifier"),
    (Intent.COMMERCIAL, _P(r"\b(best|top( \d+)?|review|reviews|rated|recommended|recommend|brands?|products?|which .* (should|to) (i )?(buy|choose|get))\b"), 0.8, "commercial modifier"),
    (Intent.PROBLEM_SOLVING, _P(r"\b(fix|fixing|problem|problems|issue|issues|not working|doesn'?t work|won'?t|stop|prevent|avoid|mistakes?|wrong|side effects?|causes?|troubleshoot|greasy|breakout|irritation|error|fail|failing|why does|why do|why is|why am|worse|clog|clogs|clogged|pores|breakouts?|sticky|white cast|pilling)\b"), 0.75, "problem phrasing"),
    (Intent.RESEARCH, _P(r"\b(statistics|stats|study|studies|research|data|evidence|science|scientific|clinical|history|trends?|facts?|myths?|ingredients?|how does .* work)\b"), 0.65, "research phrasing"),
    (Intent.NAVIGATIONAL, _P(r"\b(login|log in|sign in|official (site|website)|customer (care|service)|contact|app|website|homepage)\b"), 0.7, "navigational term"),
    (Intent.INFORMATIONAL, _P(r"^(what|why|how|when|where|who|which|can|does|do|is|are|should|will|could)\b|\b(guide|tips|meaning|definition|explained|tutorial|benefits|uses|how to|ways to|ideas)\b"), 0.7, "question / informational phrasing"),
]

FUNNEL = {
    Intent.INFORMATIONAL: "TOFU",
    Intent.RESEARCH: "TOFU",
    Intent.PROBLEM_SOLVING: "TOFU",
    Intent.COMMERCIAL: "MOFU",
    Intent.COMPARISON: "MOFU",
    Intent.TRANSACTIONAL_INVESTIGATION: "MOFU",
    Intent.TRANSACTIONAL: "BOFU",
    Intent.LOCAL: "BOFU",
    Intent.NAVIGATIONAL: "BOFU",
}

JOURNEY = {
    Intent.INFORMATIONAL: "Awareness",
    Intent.RESEARCH: "Awareness",
    Intent.PROBLEM_SOLVING: "Problem recognition",
    Intent.COMMERCIAL: "Consideration",
    Intent.COMPARISON: "Evaluation",
    Intent.TRANSACTIONAL_INVESTIGATION: "Evaluation",
    Intent.TRANSACTIONAL: "Purchase",
    Intent.LOCAL: "Purchase",
    Intent.NAVIGATIONAL: "Retention / navigation",
}

BUSINESS_WEIGHT = {
    Intent.TRANSACTIONAL: 1.0,
    Intent.LOCAL: 0.95,
    Intent.TRANSACTIONAL_INVESTIGATION: 0.9,
    Intent.COMMERCIAL: 0.85,
    Intent.COMPARISON: 0.8,
    Intent.PROBLEM_SOLVING: 0.6,
    Intent.INFORMATIONAL: 0.5,
    Intent.RESEARCH: 0.4,
    Intent.NAVIGATIONAL: 0.3,
}


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    signal: str
    secondary: list[Intent]
    funnel_stage: str
    journey: str
    query_type: str


def query_type(query: str) -> str:
    q = normalize_query(query)
    words = q.split()
    first = words[0] if words else ""
    if re.search(r"\bvs\b|\bversus\b|difference between|compared", q):
        return "comparison"
    if q.startswith("how to") or q.startswith("how do i") or q.startswith("how can i"):
        return "how-to"
    if re.search(r"\balternatives?\b|instead of", q):
        return "alternatives"
    if first in {"what", "why", "how", "when", "where", "who", "which", "can", "does", "do", "is", "are", "should", "will"}:
        return "conversational question" if len(words) >= 7 else "question"
    if re.search(r"\b(best|top)\b", q):
        return "best-of / list"
    if re.search(r"\b(price|cost|buy|online|cheap)\b", q):
        return "commercial modifier"
    if re.search(r"\bnear me\b|\bin [a-z]+$", q):
        return "local"
    if len(words) >= 5:
        return "long-tail"
    return "head / mid-tail"


def classify_intent(query: str) -> IntentResult:
    q = normalize_query(query)
    matches: list[tuple[Intent, float, str]] = []
    for intent, pattern, conf, label in RULES:
        if pattern.search(q):
            matches.append((intent, conf, label))
    if not matches:
        intent, conf, signal = Intent.INFORMATIONAL, 0.4, "no explicit modifier (default informational)"
        secondary: list[Intent] = []
    else:
        # Questions starting with what/how/why that also mention "best" are commercial-investigation
        # only when the commercial modifier dominates; rule order already encodes precedence.
        intent, conf, signal = matches[0]
        secondary = [m[0] for m in matches[1:] if m[0] != intent]
        # "how to choose the best X" -> commercial (decision research), keep informational as secondary
        if intent == Intent.PROBLEM_SOLVING and any(m[0] == Intent.COMPARISON for m in matches):
            intent, conf, signal = Intent.COMPARISON, 0.75, "comparison of problem solutions"
        # "how to get the best results…", "how often…": the how-to frame dominates a commercial modifier
        if re.match(r"^(how (to|do|can|should|often|much|many|long)|what (is|are|does)|when)\b", q) and intent in (
            Intent.COMMERCIAL, Intent.TRANSACTIONAL_INVESTIGATION
        ) and not re.search(r"\b(buy|price|cost|brands?)\b", q):
            secondary = [intent] + secondary
            intent, conf, signal = Intent.INFORMATIONAL, 0.7, "question frame dominates commercial modifier"
        # Multiple competing strong signals lower our confidence
        if len({m[0] for m in matches}) >= 3:
            conf -= 0.15
    return IntentResult(
        intent=intent,
        confidence=round(max(0.1, min(conf, 1.0)), 2),
        signal=signal,
        secondary=secondary,
        funnel_stage=FUNNEL[intent],
        journey=JOURNEY[intent],
        query_type=query_type(query),
    )
