"""Build the tabular views (pandas DataFrames) used by the CSV and XLSX exporters."""

from __future__ import annotations

import pandas as pd

from ..models import Coverage, QueryCandidate, Risk
from ..results import ResearchResult

FINAL_COLUMNS = [
    "query_id", "fan_out_query", "seed_keyword", "parent_topic", "topic_cluster", "sub_cluster", "search_volume",
    "competition", "competition_index", "cpc_low", "cpc_high", "intent", "query_type", "funnel_stage",
    "business_relevance", "ai_search_relevance", "source_frequency", "serp_evidence", "source_google", "source_paa",
    "source_reddit", "source_forums", "source_competitors", "source_llm", "existing_page", "existing_page_url",
    "coverage_status", "cannibalization_risk", "recommended_action", "recommended_content_type", "recommended_url",
    "content_gap", "fanout_score", "confidence", "reasoning",
]
EXTRA_COLUMNS = ["data_origin_label", "search_volume_source", "user_journey_stage", "origins", "evidence"]


def _yn(flag: bool) -> str:
    return "YES" if flag else "NO"


def query_row(idx: int, c: QueryCandidate, seed: str) -> dict:
    m = c.metrics
    o = c.origins
    return {
        "query_id": f"Q{idx:03d}",
        "fan_out_query": c.query,
        "seed_keyword": seed,
        "parent_topic": c.parent_topic,
        "topic_cluster": c.cluster,
        "sub_cluster": c.sub_cluster,
        "search_volume": m.search_volume if m else None,
        "competition": m.competition if m else None,
        "competition_index": m.competition_index if m else None,
        "cpc_low": m.cpc_low if m else None,
        "cpc_high": m.cpc_high if m else None,
        "intent": c.intent.value if c.intent else None,
        "query_type": c.query_type,
        "funnel_stage": c.funnel_stage,
        "business_relevance": c.business_relevance,
        "ai_search_relevance": c.ai_search_relevance,
        "source_frequency": c.source_frequency,
        "serp_evidence": c.serp_evidence,
        "source_google": _yn(bool(o & {"google", "related", "autocomplete"})),
        "source_paa": _yn("paa" in o),
        "source_reddit": _yn("reddit" in o),
        "source_forums": _yn(bool(o & {"forums", "quora"})),
        "source_competitors": _yn("competitors" in o),
        "source_llm": _yn("llm" in o),
        "existing_page": _yn(c.coverage in (Coverage.EXISTING_PAGE, Coverage.PARTIALLY_COVERED, Coverage.CANNIBALIZATION_RISK)),
        "existing_page_url": c.existing_page_url,
        "coverage_status": c.coverage.value if c.coverage else None,
        "cannibalization_risk": c.cannibalization_risk.value,
        "recommended_action": c.action.value if c.action else None,
        "recommended_content_type": c.content_type,
        "recommended_url": c.recommended_url,
        "content_gap": c.content_gap,
        "fanout_score": c.score,
        "confidence": c.confidence,
        "reasoning": c.reasoning,
        "data_origin_label": c.label.value,
        "search_volume_source": f"{m.label.value}: {m.note}" if m else "UNAVAILABLE",
        "user_journey_stage": c.user_journey,
        "origins": ", ".join(sorted(o)),
        "evidence": " | ".join(c.evidence[:5]),
    }


def final_df(r: ResearchResult) -> pd.DataFrame:
    rows = [query_row(i + 1, c, r.seed) for i, c in enumerate(r.final)]
    return pd.DataFrame(rows, columns=FINAL_COLUMNS + EXTRA_COLUMNS)


def inventory_df(r: ResearchResult) -> pd.DataFrame:
    page_by_url = {(p.final_url or p.url): p for p in r.pages}
    rows = []
    for p in r.pages:
        url = p.final_url or p.url
        inv = next((i for i in r.inventory if i.url == url), None)
        rows.append({
            "url": url, "http_status": p.status, "canonical_url": p.canonical, "indexable": _yn(p.indexable),
            "index_reason": p.index_reason, "robots_meta": p.robots_meta, "page_type": p.page_type.value,
            "title": p.title, "meta_description": p.meta_description, "h1": " | ".join(p.h1),
            "h2": " | ".join(p.h2[:20]), "h3": " | ".join(p.h3[:20]), "word_count": p.word_count,
            "primary_topic": inv.primary_topic if inv else "", "secondary_topics": " | ".join(inv.secondary_topics[:10]) if inv else "",
            "entities": ", ".join(inv.entities[:12]) if inv else "", "products": ", ".join(inv.products[:10]) if inv else "",
            "services": ", ".join(inv.services) if inv else "", "locations": ", ".join(inv.locations) if inv else "",
            "questions_answered": " | ".join(inv.questions_answered[:15]) if inv else "",
            "search_intent": inv.search_intent.value if inv else "", "likely_target_keywords": ", ".join(inv.likely_keywords) if inv else "",
            "semantic_topic": inv.semantic_topic if inv else "", "content_type": inv.content_type if inv else "",
            "funnel_stage": inv.funnel_stage if inv else "", "internal_inlinks": inv.inlinks if inv else None,
            "breadcrumbs": " > ".join(p.breadcrumbs), "schema_types": ", ".join(p.schema_types),
            "faq_questions": " | ".join(p.faq_questions[:15]), "internal_links": len(p.internal_links),
            "external_links": len(p.external_links), "image_alt_texts": " | ".join(p.image_alts[:15]),
            "product_info": str(p.product_info) if p.product_info else "", "category_info": str(p.category_info) if p.category_info else "",
        })
    _ = page_by_url
    return pd.DataFrame(rows)


def cannibalization_df(r: ResearchResult) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "risk": p.risk.value, "page_a": p.url_a, "page_b": p.url_b, "topic_similarity": p.topic_similarity,
            "intent_similarity": p.intent_similarity, "title_similarity": p.title_similarity,
            "h1_similarity": p.h1_similarity, "content_similarity": p.content_similarity,
            "entity_overlap": p.entity_overlap, "keyword_overlap": p.keyword_overlap, "combined_score": p.combined_score,
            "overlapping_topic": p.overlapping_topic, "overlapping_intent": p.overlapping_intent, "why": p.why,
            "stronger_page": p.stronger_url, "stronger_page_signals": p.stronger_reason, "recommended_action": p.recommendation,
        }
        for p in r.pairs
    ], columns=["risk", "page_a", "page_b", "topic_similarity", "intent_similarity", "title_similarity", "h1_similarity",
                "content_similarity", "entity_overlap", "keyword_overlap", "combined_score", "overlapping_topic",
                "overlapping_intent", "why", "stronger_page", "stronger_page_signals", "recommended_action"])


def content_gaps_df(r: ResearchResult) -> pd.DataFrame:
    comp_gaps = r.competitors.gaps if r.competitors else []
    rows = []
    pool = r.final if r.final else []
    for c in pool:
        if c.coverage not in (Coverage.CONTENT_GAP, Coverage.PARTIALLY_COVERED):
            continue
        comp = [g for g in comp_gaps if g.topic.lower() in c.query or c.query in g.topic.lower()]
        comp_cov = "competitors cover it" if ("competitors" in c.origins or comp) else ("no competitor data" if not comp_gaps else "not observed in competitor headings")
        rows.append({
            "query": c.query, "search_volume": c.metrics.search_volume if c.metrics else None,
            "intent": c.intent.value if c.intent else None, "existing_coverage": c.coverage.value if c.coverage else None,
            "existing_page_url": c.existing_page_url, "competitor_coverage": comp_cov,
            "opportunity_score": c.score, "opportunity": c.content_gap, "recommended_action": c.action.value if c.action else None,
            "recommended_page_type": c.content_type, "recommended_url": c.recommended_url,
            "potential_internal_links": " | ".join(f"{l['source_url']} -> '{l['suggested_anchor']}'" for l in c.internal_links),
        })
    rows.sort(key=lambda x: -(x["opportunity_score"] or 0))
    return pd.DataFrame(rows, columns=["query", "search_volume", "intent", "existing_coverage", "existing_page_url",
                                       "competitor_coverage", "opportunity_score", "opportunity", "recommended_action",
                                       "recommended_page_type", "recommended_url", "potential_internal_links"])


def topic_clusters_df(r: ResearchResult) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "parent_topic": t.parent_topic, "topic_cluster": t.cluster, "queries": t.queries, "existing_page": t.existing_page,
            "covered_topics": " | ".join(t.covered_topics), "missing_topics": " | ".join(t.missing_topics),
            "overlapping_pages": " | ".join(t.overlapping_pages), "content_gap": t.content_gap,
            "potential_new_page": t.potential_new_page, "intent_mix": t.intents,
        }
        for t in r.topic_map
    ])


def search_volume_df(r: ResearchResult) -> pd.DataFrame:
    rows = []
    for c in r.candidates:
        m = c.metrics
        rows.append({
            "query": c.query, "selected_in_final": _yn(c in r.final), "search_volume": m.search_volume if m else None,
            "competition": m.competition if m else None, "competition_index": m.competition_index if m else None,
            "cpc_low (top-of-page bid low)": m.cpc_low if m else None, "cpc_high (top-of-page bid high)": m.cpc_high if m else None,
            "monthly_history": "; ".join(f"{x['year']}-{x['month']}:{x['searches']}" for x in (m.monthly if m else [])[-12:]),
            "data_label": m.label.value if m else "UNAVAILABLE", "source": m.source if m else "unavailable",
            "note": m.note if m else "",
        })
    return pd.DataFrame(rows)


def serp_df(r: ResearchResult) -> pd.DataFrame:
    rows = []
    if r.serp:
        for s in r.serp.serps:
            for o in s.organic:
                rows.append({"query": s.query, "type": "organic", "position": o.position, "url": o.url, "title": o.title,
                             "snippet": o.snippet, "provider": s.provider})
            if s.featured_snippet:
                rows.append({"query": s.query, "type": "featured_snippet", "position": 0, "url": s.featured_snippet.get("link"),
                             "title": s.featured_snippet.get("title"), "snippet": s.featured_snippet.get("snippet"), "provider": s.provider})
            for p in s.paa:
                rows.append({"query": s.query, "type": "people_also_ask", "position": None, "url": p.get("link"),
                             "title": p.get("question"), "snippet": p.get("snippet"), "provider": s.provider})
            for rel in s.related_searches:
                rows.append({"query": s.query, "type": "related_search", "position": None, "url": None, "title": rel,
                             "snippet": None, "provider": s.provider})
        for base, sugg in r.serp.autocomplete.items():
            for sug in sugg:
                rows.append({"query": base, "type": "autocomplete", "position": None, "url": None, "title": sug,
                             "snippet": None, "provider": r.serp.provider})
    if not rows:
        rows.append({"query": None, "type": "NO DATA", "position": None, "url": None,
                     "title": (r.serp.note if r.serp else "SERP research not run"), "snippet": None, "provider": None})
    return pd.DataFrame(rows)


def community_df(r: ResearchResult) -> pd.DataFrame:
    rows = []
    if r.community:
        for q in r.community.questions:
            rows.append({"question": q.question, "category": q.category, "frequency_independent_threads": q.frequency,
                         "platforms": ", ".join(q.sources), "confidence": q.confidence, "example_urls": " | ".join(q.urls[:3]),
                         "note": "User language / need signal – not an authoritative fact"})
        if r.community.terminology:
            rows.append({"question": "USER TERMINOLOGY: " + ", ".join(r.community.terminology), "category": "terminology",
                         "frequency_independent_threads": None, "platforms": "", "confidence": "", "example_urls": "", "note": ""})
    if r.llm:
        for ins in r.llm.insights:
            rows.append({"question": ins.text, "category": f"AI search ({ins.category})", "frequency_independent_threads": None,
                         "platforms": f"{ins.provider}:{ins.model}", "confidence": "LLM_GENERATED",
                         "example_urls": " | ".join(ins.citations[:3]), "note": "LLM-generated – shows AI-search framing, not demand"})
        for ent in r.llm.entities:
            rows.append({"question": f"ENTITY: {ent['entity']} – {ent['relationship']}", "category": "AI search entity",
                         "frequency_independent_threads": None, "platforms": ent["provider"], "confidence": "LLM_GENERATED",
                         "example_urls": "", "note": ""})
    if not rows:
        note = (r.community.note if r.community else "") or "No community data collected"
        rows.append({"question": "NO DATA", "category": "", "frequency_independent_threads": None, "platforms": "",
                     "confidence": "", "example_urls": "", "note": note})
    return pd.DataFrame(rows)


def competitor_gaps_df(r: ResearchResult) -> pd.DataFrame:
    rows = []
    if r.competitors:
        for g in r.competitors.gaps:
            rows.append({"competitor_topic": g.topic, "kind": g.gap_kind, "competitors_covering": g.competitor_count,
                         "competitor_urls": " | ".join(g.competitor_urls[:5]), "site_status": g.status,
                         "best_site_page": g.best_site_url, "site_similarity": g.site_similarity,
                         "site_pages_covering": " | ".join(g.site_urls_covering)})
        for p in r.competitors.pages:
            rows.append({"competitor_topic": f"PAGE: {p.title or p.url}", "kind": "competitor page",
                         "competitors_covering": None, "competitor_urls": p.url,
                         "site_status": "analysed" if p.fetched else f"not fetched: {p.error}",
                         "best_site_page": None, "site_similarity": None,
                         "site_pages_covering": f"words={p.word_count}; H2={len(p.h2)}; FAQ={len(p.faq)}; stats={len(p.statistics)}; "
                                                f"entities={', '.join(p.entities[:8])}"})
    if not rows:
        rows.append({"competitor_topic": "NO DATA", "kind": "", "competitors_covering": None, "competitor_urls": "",
                     "site_status": r.competitors.note if r.competitors else "not run", "best_site_page": None,
                     "site_similarity": None, "site_pages_covering": ""})
    return pd.DataFrame(rows)


def query_sources_df(r: ResearchResult) -> pd.DataFrame:
    rows = []
    for c in r.candidates:
        rows.append({"query": c.query, "status": "SELECTED" if c in r.final else ("REJECTED" if c.rejected_reason else "NOT SELECTED"),
                     "rejected_reason": c.rejected_reason, "origins": ", ".join(sorted(c.origins)),
                     "data_label": c.label.value, "dimension": c.dimension, "evidence": " | ".join(c.evidence),
                     "score": c.score})
    return pd.DataFrame(rows)


def actions_df(r: ResearchResult) -> pd.DataFrame:
    rows = []
    for c in r.final:
        rows.append({"query": c.query, "recommended_action": c.action.value if c.action else None,
                     "target_url": c.recommended_url, "content_type": c.content_type, "coverage": c.coverage.value if c.coverage else None,
                     "cannibalization_risk": c.cannibalization_risk.value, "score": c.score, "why": c.content_gap})
    for p in r.pairs:
        if p.risk in (Risk.HIGH, Risk.MEDIUM):
            rows.append({"query": f"[cannibalization] {p.overlapping_topic}", "recommended_action": "CANNIBALIZATION_REVIEW",
                         "target_url": p.stronger_url, "content_type": "Consolidation", "coverage": None,
                         "cannibalization_risk": p.risk.value, "score": None,
                         "why": f"{p.url_a} vs {p.url_b}: {p.recommendation}"})
    for link in r.internal_links:
        rows.append({"query": link["query"], "recommended_action": "INTERNAL_LINK", "target_url": link["target_url"],
                     "content_type": f"link from {link['source_url']}", "coverage": None, "cannibalization_risk": None,
                     "score": None, "why": f"anchor/topic: '{link['suggested_anchor']}' – {link['reason']}"})
    return pd.DataFrame(rows)


def all_sheets(r: ResearchResult) -> dict[str, pd.DataFrame]:
    return {
        "FINAL 100": final_df(r),
        "WEBSITE INVENTORY": inventory_df(r),
        "CANNIBALIZATION": cannibalization_df(r),
        "CONTENT GAPS": content_gaps_df(r),
        "TOPIC CLUSTERS": topic_clusters_df(r),
        "SEARCH VOLUME": search_volume_df(r),
        "SERP RESEARCH": serp_df(r),
        "COMMUNITY RESEARCH": community_df(r),
        "COMPETITOR GAPS": competitor_gaps_df(r),
        "QUERY SOURCES": query_sources_df(r),
        "RECOMMENDED ACTIONS": actions_df(r),
    }
