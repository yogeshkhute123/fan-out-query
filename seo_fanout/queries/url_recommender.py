"""URL recommendations that follow the website's existing architecture."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from ..models import Action, PageInventory, PageType
from ..textutil import slugify

EDITORIAL = (PageType.BLOG, PageType.GUIDE)


class SiteArchitecture:
    def __init__(self, inventory: list[PageInventory], base_url: str):
        self.base_url = base_url.rstrip("/")
        self.existing_paths = {urlsplit(i.url).path.rstrip("/") for i in inventory}
        slash = [urlsplit(i.url).path.endswith("/") for i in inventory if urlsplit(i.url).path not in ("", "/")]
        self.trailing_slash = sum(slash) > len(slash) / 2 if slash else False
        self.prefixes: dict[PageType, Counter] = defaultdict(Counter)
        self.type_counts: Counter = Counter()
        for i in inventory:
            path = urlsplit(i.url).path
            segments = [s for s in path.split("/") if s]
            self.type_counts[i.page_type] += 1
            if len(segments) >= 2:
                self.prefixes[i.page_type]["/" + segments[0] + "/"] += 1
            elif len(segments) == 1 and i.page_type in (PageType.OTHER, PageType.LANDING):
                self.prefixes[i.page_type]["/"] += 1

    def prefix_for(self, page_type: PageType) -> tuple[str, str]:
        """Return (prefix, explanation)."""
        counts = self.prefixes.get(page_type)
        if counts:
            prefix, n = counts.most_common(1)[0]
            return prefix, f"follows the existing {page_type.value} structure ({prefix}, {n} page(s))"
        if page_type in EDITORIAL:
            for other in EDITORIAL:
                if self.prefixes.get(other):
                    prefix, n = self.prefixes[other].most_common(1)[0]
                    return prefix, f"no {page_type.value} section found; uses the existing {other.value} section {prefix}"
            if self.prefixes.get(PageType.OTHER):
                prefix, _ = self.prefixes[PageType.OTHER].most_common(1)[0]
                return prefix, "no blog/guide section detected; mirrors the site's general content URLs"
            return "/blog/", "no editorial section exists on the site – a /blog/ section would need to be created"
        if page_type == PageType.COLLECTION and self.prefixes.get(PageType.CATEGORY):
            return self.prefix_for(PageType.CATEGORY)
        if page_type == PageType.CATEGORY and self.prefixes.get(PageType.COLLECTION):
            return self.prefix_for(PageType.COLLECTION)
        return "/", f"no existing {page_type.value} section; root-level URL suggested"

    @property
    def is_ecommerce(self) -> bool:
        return self.type_counts[PageType.PRODUCT] + self.type_counts[PageType.CATEGORY] + self.type_counts[PageType.COLLECTION] >= 2

    @property
    def is_service_business(self) -> bool:
        return self.type_counts[PageType.SERVICE] >= 1

    def recommend(self, action: Action, query: str, existing_url: str | None) -> tuple[str | None, str]:
        if action in (Action.EXISTING_PAGE_OPTIMIZATION, Action.EXISTING_PAGE_EXPANSION,
                      Action.CANNIBALIZATION_REVIEW, Action.NO_ACTION):
            return existing_url, "existing page"
        if action == Action.FAQ_SECTION and existing_url:
            return existing_url + "#faq", "add as an FAQ entry on the existing page"
        page_type = {
            Action.CATEGORY_PAGE: PageType.CATEGORY,
            Action.PRODUCT_PAGE: PageType.PRODUCT,
            Action.SERVICE_PAGE: PageType.SERVICE,
            Action.COMPARISON_PAGE: PageType.BLOG,
            Action.NEW_PAGE: PageType.BLOG,
            Action.FAQ_SECTION: PageType.FAQ,
        }.get(action, PageType.BLOG)
        if page_type == PageType.BLOG and self.type_counts[PageType.GUIDE] > self.type_counts[PageType.BLOG]:
            page_type = PageType.GUIDE
        if page_type == PageType.CATEGORY and not self.prefixes.get(PageType.CATEGORY) and self.prefixes.get(PageType.COLLECTION):
            page_type = PageType.COLLECTION
        prefix, why = self.prefix_for(page_type)
        slug_source = query
        if page_type in (PageType.CATEGORY, PageType.COLLECTION, PageType.PRODUCT):
            # listing/product URLs name the thing, not the shopping modifier
            slug_source = re.sub(r"\b(buy|price|prices|cost|cheap|affordable|online|best|top|reviews?|shop|deals?)\b", " ", query)
        slug = slugify(slug_source)
        path = prefix.rstrip("/") + "/" + slug
        n = 2
        while path.rstrip("/") in self.existing_paths:
            path = prefix.rstrip("/") + f"/{slug}-{n}"
            n += 1
        if self.trailing_slash:
            path += "/"
        return self.base_url + path, why
