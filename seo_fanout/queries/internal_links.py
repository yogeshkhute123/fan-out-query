"""Internal linking opportunities with natural (non exact-match) anchors."""

from __future__ import annotations

from ..analysis.embeddings import SemanticModel
from ..models import Action, PageData, PageInventory, QueryCandidate
from ..textutil import normalize_query

LINK_ACTIONS = {
    Action.NEW_PAGE, Action.COMPARISON_PAGE, Action.CATEGORY_PAGE, Action.PRODUCT_PAGE, Action.SERVICE_PAGE,
    Action.EXISTING_PAGE_EXPANSION, Action.EXISTING_PAGE_OPTIMIZATION, Action.FAQ_SECTION,
}


def _anchor_variants(c: QueryCandidate, target_inv: PageInventory | None) -> list[str]:
    q = normalize_query(c.query)
    topic = c.sub_cluster.split(" · ")[0] if c.sub_cluster else ""
    variants = [q]
    if target_inv is not None and target_inv.primary_topic and target_inv.primary_topic != q:
        variants.append(target_inv.primary_topic)
    if topic and topic not in ("core topic",):
        variants.append(f"{c.parent_topic} {topic}".strip())
    variants.append(f"our guide on {c.parent_topic}" if c.parent_topic else q)
    return list(dict.fromkeys(variants))


def find_internal_links(
    cands: list[QueryCandidate], inventory: list[PageInventory], pages: list[PageData], model: SemanticModel,
    per_query: int = 3, min_similarity: float = 0.2,
) -> list[dict[str, str]]:
    if not cands or not inventory:
        return []
    links_from = {(p.final_url or p.url).rstrip("/"): {u.rstrip("/") for u in p.internal_links} for p in pages}
    inv_by_url = {i.url: i for i in inventory}
    page_texts = [" . ".join([i.primary_topic, i.title] + i.secondary_topics[:10]) for i in inventory]
    sim = SemanticModel.cosine_matrix(model.encode([c.query for c in cands]), model.encode(page_texts))

    rows: list[dict[str, str]] = []
    for row, c in enumerate(cands):
        if c.action not in LINK_ACTIONS or not c.recommended_url:
            continue
        target = c.recommended_url.split("#")[0].rstrip("/")
        target_inv = inv_by_url.get(target) or inv_by_url.get(target + "/")
        anchors = _anchor_variants(c, target_inv)
        order = sim[row].argsort()[::-1]
        picked = 0
        c.internal_links = []
        for j in order:
            if picked >= per_query or sim[row, j] < min_similarity:
                break
            src = inventory[j].url
            if src.rstrip("/") == target:
                continue
            if target in links_from.get(src.rstrip("/"), set()):
                continue  # already links there
            anchor = anchors[picked % len(anchors)]
            link = {
                "query": c.query,
                "source_url": src,
                "target_url": c.recommended_url,
                "suggested_anchor": anchor,
                "reason": f"Source page is topically related (similarity {sim[row, j]:.2f}: '{inventory[j].primary_topic}') "
                          f"and does not yet link to the target.",
            }
            c.internal_links.append(link)
            rows.append(link)
            picked += 1
    return rows
