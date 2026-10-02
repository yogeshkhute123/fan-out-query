"""Small, dependency-free text helpers (normalisation, tokens, n-grams, slugs)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

STOPWORDS = frozenset(
    """
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's
where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves also get got just us will one may use using used really
""".split()
)

QUESTION_WORDS = (
    "what", "why", "how", "when", "where", "who", "which", "can", "does", "do", "is", "are",
    "should", "will", "could", "would", "did", "has", "have",
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_BOILER_SPLIT = re.compile(r"\s+[|\-–—:·•]\s+")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_query(text: str) -> str:
    """Lowercase, strip punctuation/accents and collapse whitespace."""
    text = strip_accents(text.lower())
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_query(text))


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def singularize(word: str) -> str:
    """Very small English singulariser – good enough for duplicate detection."""
    if len(word) <= 3 or word.endswith("ss") or word.endswith("us") or word.endswith("is"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def stem_tokens(text: str) -> list[str]:
    return [singularize(t) for t in content_tokens(text)]


def permutation_signature(text: str) -> str:
    """Order-independent signature: same bag of content words -> same signature."""
    return " ".join(sorted(set(stem_tokens(text))))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ngrams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def key_phrases(text: str, top_k: int = 15, max_n: int = 3) -> list[str]:
    """Frequency-based key phrase extraction (RAKE-like, no external model).

    Phrases are built only from runs of non-stopwords, so they read naturally.
    """
    runs: list[list[str]] = []
    current: list[str] = []
    for tok in tokens(text):
        if tok in STOPWORDS or len(tok) < 2 or tok.isdigit():
            if current:
                runs.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        runs.append(current)

    counts: Counter[str] = Counter()
    for run in runs:
        for n in range(1, max_n + 1):
            for gram in ngrams(run, n):
                counts[gram] += 1
    # Prefer multi-word phrases: weight = freq * (1 + 0.6*(n-1))
    scored = {
        phrase: freq * (1 + 0.6 * (phrase.count(" ")))
        for phrase, freq in counts.items()
        if freq >= 2 or phrase.count(" ") >= 1
    }
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    result: list[str] = []
    for phrase, _ in ranked:
        if any(phrase in kept for kept in result):
            continue
        result.append(phrase)
        if len(result) >= top_k:
            break
    return result


def strip_brand_suffix(title: str) -> str:
    """'Best Sunscreen for Oily Skin | Brand' -> 'Best Sunscreen for Oily Skin'.

    The trailing segment of a separated title is almost always the site/brand
    name, so it is dropped; descriptive middle segments are kept.
    """
    title = clean_text(title)
    parts = [p.strip() for p in _BOILER_SPLIT.split(title) if p.strip()]
    if len(parts) > 1:
        return " - ".join(parts[:-1])
    return title


# Light synonym canonicalisation used only for similarity (not for display).
SYNONYMS = {
    "top": "best", "finest": "best", "greatest": "best", "versus": "vs", "purchase": "buy", "shop": "buy",
    "pricing": "price", "cost": "price", "costs": "price", "cheapest": "cheap", "inexpensive": "cheap",
    "budget": "cheap", "affordable": "cheap", "suggestion": "recommendation", "recommended": "recommendation",
    "recommend": "recommendation", "tips": "tip", "guidelines": "guide", "reapply": "apply", "reapplying": "apply",
    "applying": "apply", "using": "use", "used": "use", "uses": "use", "choosing": "choose", "picking": "choose",
    "pick": "choose", "select": "choose", "greasy": "oily", "greasiness": "oily", "oiliness": "oily",
}


def canonical_tokens(text: str) -> list[str]:
    return [SYNONYMS.get(w, w) for w in (singularize(t) for t in tokens(text))]


def canonical_text(text: str) -> str:
    """Lower-cased, singularised, synonym-normalised text for similarity models."""
    return " ".join(canonical_tokens(text))


SKLEARN_STOPWORDS = sorted({w for s in STOPWORDS for w in re.findall(r"[a-z0-9]+", s)})


def is_question(text: str) -> bool:
    t = normalize_query(text)
    if text.strip().endswith("?"):
        return True
    first = t.split(" ", 1)[0] if t else ""
    return first in QUESTION_WORDS


def slugify(text: str, max_words: int = 7) -> str:
    keep = {"vs", "for", "to", "why", "how", "what", "when", "which", "best", "without", "not"}
    words = [w.replace("'", "") for w in tokens(text) if w not in STOPWORDS or w in keep]
    # Trim leading glue words
    while words and words[0] in {"for", "to", "vs"}:
        words.pop(0)
    slug = "-".join(words[:max_words])
    return re.sub(r"-+", "-", slug).strip("-") or "page"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def extract_question_sentences(text: str, limit: int = 20) -> list[str]:
    """Pull question sentences out of free text (titles, posts, snippets)."""
    found: list[str] = []
    for sentence in re.split(r"(?<=[?.!])\s+|\n+", text or ""):
        s = clean_text(sentence)
        if 12 <= len(s) <= 180 and s.endswith("?"):
            found.append(s)
        if len(found) >= limit:
            break
    return found
