"""Container for everything a research run produces (consumed by exporters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .analysis.topics import TopicMapRow
from .models import CannibalizationPair, PageData, PageInventory, QueryCandidate
from .research.community import CommunityResearch
from .research.competitors import CompetitorResearch
from .research.keyword_metrics import MetricsResult
from .research.llm_research import LLMResearch
from .research.serp_research import SerpResearch


@dataclass
class ResearchResult:
    run_id: str
    seed: str
    domain: str
    base_url: str
    country: str
    language: str
    requested: int
    pages: list[PageData] = field(default_factory=list)
    inventory: list[PageInventory] = field(default_factory=list)
    pairs: list[CannibalizationPair] = field(default_factory=list)
    serp: SerpResearch | None = None
    community: CommunityResearch | None = None
    competitors: CompetitorResearch | None = None
    llm: LLMResearch | None = None
    metrics: MetricsResult | None = None
    candidates: list[QueryCandidate] = field(default_factory=list)  # after dedup, incl. rejected
    rejected: list[QueryCandidate] = field(default_factory=list)
    final: list[QueryCandidate] = field(default_factory=list)
    topic_map: list[TopicMapRow] = field(default_factory=list)
    internal_links: list[dict[str, str]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    provider_status: dict[str, bool] = field(default_factory=dict)
    data_notes: list[str] = field(default_factory=list)
    semantic_backend: str = ""
    crawl_reused: bool = False
