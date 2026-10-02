"""Country / language code mapping for SERP and Google Ads providers.

Google Ads criterion IDs come from Google's published geo-target and language
constant tables. Unknown countries fall back to Google Ads'
GeoTargetConstantService lookup (when configured) or can be overridden with
GOOGLE_ADS_GEO_TARGET_ID / GOOGLE_ADS_LANGUAGE_ID.
"""

from __future__ import annotations

import os

# country name -> (ISO-3166 alpha-2 for SERP 'gl', Google Ads geo target constant ID, google domain)
COUNTRIES: dict[str, tuple[str, int, str]] = {
    "india": ("in", 2356, "google.co.in"),
    "united states": ("us", 2840, "google.com"),
    "usa": ("us", 2840, "google.com"),
    "us": ("us", 2840, "google.com"),
    "united kingdom": ("uk", 2826, "google.co.uk"),
    "uk": ("uk", 2826, "google.co.uk"),
    "canada": ("ca", 2124, "google.ca"),
    "australia": ("au", 2036, "google.com.au"),
    "germany": ("de", 2276, "google.de"),
    "france": ("fr", 2250, "google.fr"),
    "spain": ("es", 2724, "google.es"),
    "italy": ("it", 2380, "google.it"),
    "brazil": ("br", 2076, "google.com.br"),
    "mexico": ("mx", 2484, "google.com.mx"),
    "japan": ("jp", 2392, "google.co.jp"),
    "singapore": ("sg", 2702, "google.com.sg"),
    "united arab emirates": ("ae", 2784, "google.ae"),
    "uae": ("ae", 2784, "google.ae"),
    "netherlands": ("nl", 2528, "google.nl"),
    "south africa": ("za", 2710, "google.co.za"),
    "new zealand": ("nz", 2554, "google.co.nz"),
    "ireland": ("ie", 2372, "google.ie"),
    "indonesia": ("id", 2360, "google.co.id"),
    "philippines": ("ph", 2608, "google.com.ph"),
    "malaysia": ("my", 2458, "google.com.my"),
    "pakistan": ("pk", 2586, "google.com.pk"),
    "nigeria": ("ng", 2566, "google.com.ng"),
    "saudi arabia": ("sa", 2682, "google.com.sa"),
}

# language name -> (ISO-639-1 'hl', Google Ads language constant ID)
LANGUAGES: dict[str, tuple[str, int]] = {
    "english": ("en", 1000),
    "german": ("de", 1001),
    "french": ("fr", 1002),
    "spanish": ("es", 1003),
    "italian": ("it", 1004),
    "japanese": ("ja", 1005),
    "dutch": ("nl", 1010),
    "korean": ("ko", 1012),
    "portuguese": ("pt", 1014),
    "chinese": ("zh-cn", 1017),
    "arabic": ("ar", 1019),
    "hindi": ("hi", 1023),
    "indonesian": ("id", 1025),
    "polish": ("pl", 1030),
    "russian": ("ru", 1031),
    "turkish": ("tr", 1037),
}

_ISO_TO_COUNTRY = {v[0]: k for k, v in COUNTRIES.items()}
_ISO_TO_LANG = {v[0]: k for k, v in LANGUAGES.items()}


def country_info(country: str) -> tuple[str | None, int | None, str]:
    key = (country or "").strip().lower()
    if key in COUNTRIES:
        gl, geo, domain = COUNTRIES[key]
    elif key in _ISO_TO_COUNTRY:
        gl, geo, domain = COUNTRIES[_ISO_TO_COUNTRY[key]]
    else:
        gl, geo, domain = (key if len(key) == 2 else None), None, "google.com"
    override = os.getenv("GOOGLE_ADS_GEO_TARGET_ID")
    if override and override.isdigit():
        geo = int(override)
    return gl, geo, domain


def language_info(language: str) -> tuple[str, int | None]:
    key = (language or "english").strip().lower()
    if key in LANGUAGES:
        hl, lang_id = LANGUAGES[key]
    elif key in _ISO_TO_LANG:
        hl, lang_id = LANGUAGES[_ISO_TO_LANG[key]]
    else:
        hl, lang_id = (key[:2] if key else "en"), None
    override = os.getenv("GOOGLE_ADS_LANGUAGE_ID")
    if override and override.isdigit():
        lang_id = int(override)
    return hl, lang_id
