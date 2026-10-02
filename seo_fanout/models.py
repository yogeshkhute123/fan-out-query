"""Typed data structures shared across the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DataLabel(str, Enum):
    """Provenance label attached to every data point that reaches the output."""

    REAL = "REAL_DATA"  # observed from a crawl or an API (Google Ads, SERP provider, Reddit)
    ESTIMATED = "ESTIMATED_DATA"  # heuristic computation over real data
    LLM = "LLM_GENERATED"  # produced by an LLM; never treated as a fact
    TEMPLATE = "TEMPLATE_GENERATED"  # produced by rule-based query patterns
    UNAVAILABLE = "UNAVAILABLE"


class PageType(str, Enum):
    HOMEPAGE = "Homepage"
    CATEGORY = "Category"
    COLLECTION = "Collection"
    PRODUCT = "Product"
    BLOG = "Blog"
    GUIDE = "Guide"
    LANDING = "Landing page"
    SERVICE = "Service page"
    LOCATION = "Location page"
    FAQ = "FAQ"
    OTHER = "Other"


class Intent(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    COMMERCIAL = "COMMERCIAL"
    TRANSACTIONAL = "TRANSACTIONAL"
    NAVIGATIONAL = "NAVIGATIONAL"
    LOCAL = "LOCAL"
    COMPARISON = "COMPARISON"
    PROBLEM_SOLVING = "PROBLEM_SOLVING"
    RESEARCH = "RESEARCH"
    TRANSACTIONAL_INVESTIGATION = "TRANSACTIONAL_INVESTIGATION"


class Coverage(str, Enum):
    EXISTING_PAGE = "EXISTING_PAGE"
    PARTIALLY_COVERED = "PARTIALLY_COVERED"
    CONTENT_GAP = "CONTENT_GAP"
    CANNIBALIZATION_RISK = "CANNIBALIZATION_RISK"


class Action(str, Enum):
    EXISTING_PAGE_OPTIMIZATION = "EXISTING_PAGE_OPTIMIZATION"
    EXISTING_PAGE_EXPANSION = "EXISTING_PAGE_EXPANSION"
    NEW_PAGE = "NEW_PAGE"
    FAQ_SECTION = "FAQ_SECTION"
    CATEGORY_PAGE = "CATEGORY_PAGE"
    PRODUCT_PAGE = "PRODUCT_PAGE"
    SERVICE_PAGE = "SERVICE_PAGE"
    COMPARISON_PAGE = "COMPARISON_PAGE"
    NO_ACTION = "NO_ACTION"
    CANNIBALIZATION_REVIEW = "CANNIBALIZATION_REVIEW"


class Risk(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Website
# ---------------------------------------------------------------------------
class PageData(BaseModel):
    url: str
    final_url: str | None = None
    status: int | None = None
    canonical: str | None = None
    indexable: bool = True
    index_reason: str = ""
    robots_meta: str = ""
    title: str = ""
    meta_description: str = ""
    h1: list[str] = Field(default_factory=list)
    h2: list[str] = Field(default_factory=list)
    h3: list[str] = Field(default_factory=list)
    main_content: str = ""
    word_count: int = 0
    image_alts: list[str] = Field(default_factory=list)
    breadcrumbs: list[str] = Field(default_factory=list)
    internal_links: list[str] = Field(default_factory=list)
    internal_anchors: dict[str, str] = Field(default_factory=dict)
    external_links: list[str] = Field(default_factory=list)
    schema_types: list[str] = Field(default_factory=list)
    faq_questions: list[str] = Field(default_factory=list)
    product_info: dict[str, Any] = Field(default_factory=dict)
    category_info: dict[str, Any] = Field(default_factory=dict)
    page_type: PageType = PageType.OTHER
    depth: int = 0
    content_hash: str = ""
    lang: str = ""


class PageInventory(BaseModel):
    url: str
    page_type: PageType
    primary_topic: str
    secondary_topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    questions_answered: list[str] = Field(default_factory=list)
    search_intent: Intent = Intent.INFORMATIONAL
    likely_keywords: list[str] = Field(default_factory=list)
    semantic_topic: str = ""
    content_type: str = ""
    funnel_stage: str = ""
    word_count: int = 0
    inlinks: int = 0
    title: str = ""
    h1: str = ""


class CannibalizationPair(BaseModel):
    url_a: str
    url_b: str
    topic_similarity: float
    intent_similarity: float
    title_similarity: float
    h1_similarity: float
    content_similarity: float
    entity_overlap: float
    keyword_overlap: float
    combined_score: float
    risk: Risk
    overlapping_topic: str = ""
    overlapping_intent: str = ""
    why: str = ""
    stronger_url: str | None = None
    stronger_reason: str = ""
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------
class OrganicResult(BaseModel):
    position: int
    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""


class SerpData(BaseModel):
    query: str
    provider: str
    organic: list[OrganicResult] = Field(default_factory=list)
    paa: list[dict[str, str]] = Field(default_factory=list)  # {question, snippet, link}
    related_searches: list[str] = Field(default_factory=list)
    featured_snippet: dict[str, str] | None = None
    autocomplete: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    error: str | None = None


class CommunityItem(BaseModel):
    source: str  # reddit | quora | forum
    platform_domain: str
    thread_url: str
    title: str
    text: str = ""
    score: int | None = None
    num_comments: int | None = None
    subreddit: str | None = None


class CommunityQuestion(BaseModel):
    question: str
    sources: list[str] = Field(default_factory=list)  # e.g. reddit, quora, forum
    urls: list[str] = Field(default_factory=list)
    frequency: int = 1
    category: str = "question"  # question|problem|objection|misconception|comparison|buying_concern
    confidence: str = "LOW"


class CompetitorPage(BaseModel):
    url: str
    domain: str
    serp_position: int | None = None
    title: str = ""
    h1: list[str] = Field(default_factory=list)
    h2: list[str] = Field(default_factory=list)
    h3: list[str] = Field(default_factory=list)
    faq: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    statistics: list[str] = Field(default_factory=list)
    word_count: int = 0
    fetched: bool = False
    error: str | None = None


class LLMInsight(BaseModel):
    provider: str
    model: str
    category: str
    text: str
    citations: list[str] = Field(default_factory=list)


class KeywordMetrics(BaseModel):
    keyword: str
    search_volume: int | None = None
    competition: str | None = None
    competition_index: int | None = None
    cpc_low: float | None = None
    cpc_high: float | None = None
    monthly: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "unavailable"
    label: DataLabel = DataLabel.UNAVAILABLE
    note: str = ""


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
class QueryCandidate(BaseModel):
    query: str
    normalized: str = ""
    origins: set[str] = Field(default_factory=set)  # google, paa, related, autocomplete, reddit, ...
    evidence: list[str] = Field(default_factory=list)
    dimension: str = ""  # generation dimension (core, howto, problems, ...)
    label: DataLabel = DataLabel.TEMPLATE

    # analysis
    intent: Intent | None = None
    intent_confidence: float = 0.0
    query_type: str = ""
    funnel_stage: str = ""
    user_journey: str = ""
    business_relevance: float = 0.0
    ai_search_relevance: float = 0.0
    topical_relevance: float = 0.0
    serp_evidence: float = 0.0

    coverage: Coverage | None = None
    existing_page_url: str | None = None
    matching_pages: list[tuple[str, float]] = Field(default_factory=list)
    cannibalization_risk: Risk = Risk.NONE
    action: Action | None = None
    content_type: str = ""
    recommended_url: str | None = None
    content_gap: str = ""

    parent_topic: str = ""
    cluster: str = ""
    sub_cluster: str = ""

    metrics: KeywordMetrics | None = None
    demand_proxy: float | None = None  # ESTIMATED from autocomplete agreement/rank – never a volume
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    confidence: str = "LOW"
    reasoning: str = ""
    rejected_reason: str | None = None
    internal_links: list[dict[str, str]] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def source_frequency(self) -> int:
        return len(self.origins)
