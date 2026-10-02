"""URL validation and normalisation."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

TRACKING_PARAMS = re.compile(
    r"^(utm_.*|gclid|fbclid|msclkid|mc_cid|mc_eid|_ga|_gl|ref|ref_src|igshid|yclid|srsltid|sessionid|sid|phpsessid)$",
    re.I,
)
NON_HTML_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|bmp|tiff|pdf|zip|rar|gz|tar|7z|mp3|mp4|avi|mov|wmv|webm|"
    r"css|js|json|xml|txt|csv|xls|xlsx|doc|docx|ppt|pptx|woff2?|ttf|eot|exe|dmg|apk)$",
    re.I,
)
SKIP_PATH_PATTERNS = re.compile(
    r"/(cart|checkout|account|login|logout|register|signin|signup|my-account|wp-admin|wp-login|"
    r"wp-json|admin|search|cdn-cgi|feed)(/|$)|/(add-to-cart|compare|wishlist)|\?.*(add-to-cart|orderby|filter_|"
    r"replytocom|share=)",
    re.I,
)
DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)([a-z0-9-]{1,63}(?<!-)\.)+[a-z]{2,63}$|^localhost$|^\d{1,3}(\.\d{1,3}){3}$", re.I
)


class InvalidDomainError(ValueError):
    pass


def validate_domain(raw: str) -> str:
    """Return a canonical base URL (scheme://host[:port]) or raise InvalidDomainError."""
    raw = (raw or "").strip()
    if not raw:
        raise InvalidDomainError("Website domain is required.")
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    parts = urlsplit(raw)
    host = (parts.hostname or "").lower()
    if not host or not DOMAIN_RE.match(host):
        raise InvalidDomainError(f"'{raw}' is not a valid domain.")
    netloc = host + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme.lower(), netloc, "", "", ""))


def site_key(host: str) -> str:
    """Host without leading www. – used to treat www/non-www as the same site."""
    host = (host or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_same_site(url: str, base_host: str) -> bool:
    host = urlsplit(url).hostname or ""
    return site_key(host) == site_key(base_host)


def normalize_url(url: str, base: str | None = None, strip_query: bool = False) -> str | None:
    """Normalise a URL for de-duplication. Returns None for unusable URLs."""
    if not url:
        return None
    url = url.strip()
    if url.startswith(("mailto:", "tel:", "javascript:", "data:", "#", "sms:", "whatsapp:")):
        return None
    if base:
        url = urljoin(base, url)
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower()
    if not host:
        return None
    port = parts.port
    netloc = host
    if port and not ((parts.scheme == "http" and port == 80) or (parts.scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    path = re.sub(r"/index\.(html?|php|aspx?)$", "/", path, flags=re.I)
    query = ""
    if not strip_query and parts.query:
        pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not TRACKING_PARAMS.match(k)]
        query = urlencode(sorted(pairs))
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def dedup_key(url: str) -> str:
    """Key that treats trailing-slash and www variants as the same page."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return f"{site_key(parts.hostname or '')}{path}?{parts.query}".lower()


def is_crawlable_path(url: str) -> bool:
    parts = urlsplit(url)
    if NON_HTML_EXT.search(parts.path):
        return False
    if SKIP_PATH_PATTERNS.search(parts.path + ("?" + parts.query if parts.query else "")):
        return False
    return True


def url_depth(url: str) -> int:
    return len([p for p in urlsplit(url).path.split("/") if p])


def is_pagination(url: str) -> bool:
    parts = urlsplit(url)
    return bool(re.search(r"/page/\d+/?$", parts.path) or re.search(r"(^|&)(page|p|pg)=\d+", parts.query))
