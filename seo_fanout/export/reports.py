"""Markdown reports: cannibalization, content gaps and the research summary."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..models import Action, Coverage, Risk
from ..results import ResearchResult


def _md_escape(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def cannibalization_report(r: ResearchResult) -> str:
    lines = [
        "# Cannibalization Report", "",
        f"**Website:** {r.base_url}  ",
        f"**Run ID:** {r.run_id}  ",
        f"**Indexable pages compared:** {len(r.inventory)}  ",
        f"**Similarity backend:** {r.semantic_backend}", "",
        "> Cannibalization here means *same search intent + overlapping topical purpose + competing page targeting*. "
        "Sharing a keyword alone is not treated as cannibalization. These are recommendations only – nothing on the "
        "website is modified. Validate HIGH-risk pairs with Google Search Console query/page data before acting.", "",
    ]
    for risk in (Risk.HIGH, Risk.MEDIUM, Risk.LOW):
        group = [p for p in r.pairs if p.risk == risk]
        lines += [f"## {risk.value.title()} Risk ({len(group)})", ""]
        if not group:
            lines += ["_None detected._", ""]
            continue
        limit = None if risk != Risk.LOW else 40
        for p in group[:limit]:
            lines += [
                f"### {p.url_a}  ↔  {p.url_b}", "",
                f"- **URL A:** {p.url_a}",
                f"- **URL B:** {p.url_b}",
                f"- **Overlap:** topic “{p.overlapping_topic}”; intent {p.overlapping_intent}",
                f"- **Why they may compete:** {p.why}",
            ]
            if risk == Risk.HIGH:
                lines.append(f"- **Stronger topical target:** {p.stronger_url} ({p.stronger_reason})")
            lines += [f"- **Recommended action:** {p.recommendation}", ""]
        if limit and len(group) > limit:
            lines += [f"_…{len(group) - limit} more LOW-risk pairs in the XLSX 'CANNIBALIZATION' sheet._", ""]

    lines += ["## Similarity Matrix", "", "| Page A | Page B | Topic Similarity | Intent Similarity | Risk |",
              "|---|---|---|---|---|"]
    for p in r.pairs[:150]:
        lines.append(f"| {p.url_a} | {p.url_b} | {p.topic_similarity:.2f} | {p.intent_similarity:.2f} | {p.risk.value} |")
    if not r.pairs:
        lines.append("| – | – | – | – | NONE |")
    lines.append("")

    risky_queries = [c for c in r.final if c.coverage == Coverage.CANNIBALIZATION_RISK]
    if risky_queries:
        lines += ["## Fan-out Queries Affected by Cannibalization", ""]
        for c in risky_queries:
            lines.append(f"- **{c.query}** – {c.content_gap}")
        lines.append("")
    return "\n".join(lines)


def content_gap_report(r: ResearchResult) -> str:
    gaps = [c for c in r.final if c.coverage in (Coverage.CONTENT_GAP, Coverage.PARTIALLY_COVERED)]
    gaps.sort(key=lambda c: -c.score)
    comp_available = bool(r.competitors and r.competitors.available)
    lines = [
        "# Content Gap Report", "",
        f"**Website:** {r.base_url}  ", f"**Seed keyword:** {r.seed}  ", f"**Run ID:** {r.run_id}", "",
        f"Gaps in the final query set: **{sum(c.coverage == Coverage.CONTENT_GAP for c in gaps)}** not covered, "
        f"**{sum(c.coverage == Coverage.PARTIALLY_COVERED for c in gaps)}** partially covered.", "",
    ]
    if r.metrics and not r.metrics.available:
        lines += [f"> Search volume: **unavailable** – {r.metrics.note}. Volumes are left empty, never estimated.", ""]
    for c in gaps:
        vol = c.metrics.search_volume if c.metrics and c.metrics.search_volume is not None else "unavailable"
        comp = ("covered by competitors" if "competitors" in c.origins
                else ("not observed in analysed competitor pages" if comp_available else "no competitor data"))
        links = "; ".join(f"{l['source_url']} (anchor/topic: “{l['suggested_anchor']}”)" for l in c.internal_links) or "none identified"
        lines += [
            f"### {c.query}", "",
            f"- **Search Volume:** {vol}",
            f"- **Intent:** {c.intent.value if c.intent else ''}",
            f"- **Existing Coverage:** {c.coverage.value if c.coverage else ''}" + (f" – {c.existing_page_url}" if c.existing_page_url else ""),
            f"- **Competitor Coverage:** {comp}",
            f"- **Opportunity:** {c.content_gap} (score {c.score}, confidence {c.confidence})",
            f"- **Recommended Page Type:** {c.content_type} ({c.action.value if c.action else ''})",
            f"- **Recommended URL:** {c.recommended_url}",
            f"- **Potential Internal Links:** {links}", "",
        ]
    if r.competitors and r.competitors.gaps:
        not_cov = [g for g in r.competitors.gaps if g.status in ("NOT_COVERED", "POORLY_COVERED")][:40]
        if not_cov:
            lines += ["## Competitor Topics Missing or Weak on the Website", "",
                      "| Competitor topic | Competitors | Site status | Closest site page |", "|---|---|---|---|"]
            for g in not_cov:
                lines.append(f"| {_md_escape(g.topic)} | {g.competitor_count} | {g.status} | {g.best_site_url or '–'} |")
            lines.append("")
        multi = [g for g in r.competitors.gaps if g.status == "COVERED_BY_MULTIPLE_PAGES"][:20]
        if multi:
            lines += ["## Topics Covered by Multiple Existing Pages (possible cannibalization)", ""]
            for g in multi:
                lines.append(f"- {g.topic}: {', '.join(g.site_urls_covering)}")
            lines.append("")
    return "\n".join(lines)


def summary_report(r: ResearchResult) -> str:
    risk = Counter(p.risk.value for p in r.pairs)
    final = r.final
    by_intent = Counter(c.intent.value for c in final if c.intent)
    by_cluster = Counter(c.cluster for c in final)
    by_type = Counter(c.content_type for c in final)
    by_cov = Counter(c.coverage.value for c in final if c.coverage)
    by_action = Counter(c.action.value for c in final if c.action)
    expand = Counter(c.existing_page_url.split("#")[0] for c in final
                     if c.action in (Action.EXISTING_PAGE_EXPANSION, Action.EXISTING_PAGE_OPTIMIZATION, Action.FAQ_SECTION)
                     and c.existing_page_url)
    new_pages = [c for c in final if c.action in (Action.NEW_PAGE, Action.COMPARISON_PAGE, Action.CATEGORY_PAGE,
                                                  Action.PRODUCT_PAGE, Action.SERVICE_PAGE)]
    gaps = sum(1 for c in r.candidates if c.coverage == Coverage.CONTENT_GAP and not c.rejected_reason)
    existing_vs_new = Counter("existing page" if c.coverage in (Coverage.EXISTING_PAGE, Coverage.PARTIALLY_COVERED,
                                                                 Coverage.CANNIBALIZATION_RISK) else "new page / gap" for c in final)

    def table(counter: Counter, header: str) -> list[str]:
        out = [f"| {header} | Queries |", "|---|---|"]
        out += [f"| {_md_escape(k)} | {v} |" for k, v in counter.most_common()]
        return out + [""]

    lines = [
        "# SEO Fan-Out Research Report", "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · run `{r.run_id}`_", "",
        "## Website", "", f"{r.base_url}", "",
        "## Seed Keyword", "", f"**{r.seed}** · country: {r.country or '–'} · language: {r.language or '–'}", "",
        "## Pages Crawled", "",
        f"{len(r.pages)} HTML pages crawled ({len(r.inventory)} indexable)" + (" – reused cached crawl" if r.crawl_reused else ""), "",
        "## Cannibalization Summary", "",
        f"- High: {risk.get('HIGH', 0)}", f"- Medium: {risk.get('MEDIUM', 0)}", f"- Low: {risk.get('LOW', 0)}", "",
        "## Content Gaps", "",
        f"{gaps} validated candidate queries are not served by any existing page "
        f"({by_cov.get('CONTENT_GAP', 0)} of them are in the final list).", "",
        "## Top Fan-Out Opportunities", "",
        "| # | Query | Score | Intent | Coverage | Action | Volume |", "|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(final[:20], 1):
        vol = c.metrics.search_volume if c.metrics and c.metrics.search_volume is not None else "n/a"
        lines.append(f"| {i} | {_md_escape(c.query)} | {c.score} | {c.intent.value if c.intent else ''} | "
                     f"{c.coverage.value if c.coverage else ''} | {c.action.value if c.action else ''} | {vol} |")
    lines += ["", "## Existing Pages That Should Be Expanded", ""]
    lines += [f"- {url} – {n} query/queries" for url, n in expand.most_common(25)] or ["_None._"]
    lines += ["", "## New Pages Worth Creating", ""]
    lines += [f"- **{c.query}** → {c.recommended_url} ({c.content_type}, score {c.score})" for c in new_pages[:30]] or ["_None._"]
    lines += ["", "## Internal Linking Opportunities", "", f"{len(r.internal_links)} suggested links."]
    for l in r.internal_links[:15]:
        lines.append(f"- {l['source_url']} → {l['target_url']} (anchor/topic: “{l['suggested_anchor']}”)")
    lines += ["", f"## Final {r.requested} Query Summary", "",
              f"**{len(final)}** queries selected (requested {r.requested}).", ""]
    if r.stats.get("selection", {}).get("shortfall_reason"):
        lines += [f"> ⚠️ {r.stats['selection']['shortfall_reason']}", ""]
    lines += ["### By intent", ""] + table(by_intent, "Intent")
    lines += ["### By topic cluster", ""] + table(by_cluster, "Cluster")
    lines += ["### By content type", ""] + table(by_type, "Content type")
    lines += ["### Existing vs new page", ""] + table(existing_vs_new, "Destination")
    lines += ["### By recommended action", ""] + table(by_action, "Action")

    lines += ["## Data Sources & Provenance", "",
              "| Provider | Status |", "|---|---|"]
    lines += [f"| {k} | {'configured' if v else 'not configured – skipped'} |" for k, v in r.provider_status.items()]
    lines += ["", "- **REAL_DATA**: crawled pages, SERP-API results, Reddit API threads, competitor pages, Google Ads metrics.",
              "- **ESTIMATED_DATA**: similarity, intent, coverage and scores (heuristics computed over real data).",
              "- **LLM_GENERATED**: AI-search questions and LLM fan-out ideas – used as hypotheses, never as facts.",
              "- **TEMPLATE_GENERATED**: rule-based fan-out patterns.",
              f"- Semantic similarity backend: `{r.semantic_backend}`.", ""]
    if r.data_notes:
        lines += ["### Notes on unavailable data", ""] + [f"- {n}" for n in r.data_notes] + [""]
    lines += ["## Pipeline Statistics", "", "```"]
    for k, v in r.stats.items():
        lines.append(f"{k}: {v}")
    lines += ["```", ""]
    return "\n".join(lines)


def write_reports(r: ResearchResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cannibalization": out_dir / "cannibalization_report.md",
        "content_gap": out_dir / "content_gap_report.md",
        "summary": out_dir / "research_summary.md",
    }
    paths["cannibalization"].write_text(cannibalization_report(r), encoding="utf-8")
    paths["content_gap"].write_text(content_gap_report(r), encoding="utf-8")
    paths["summary"].write_text(summary_report(r), encoding="utf-8")
    return paths
