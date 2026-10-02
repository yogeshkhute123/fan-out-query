# SEO Fan-Out Query Research & Cannibalization Analysis

A cannibalization-aware SEO / GEO research tool. It does **not** blindly generate 100 keyword variations.
It first crawls and understands the website, then researches demand (SERPs, People Also Ask, Reddit/forums,
competitors, AI assistants) and finally decides **which page should serve each search need**.

You provide a **seed keyword** and a **website**. The tool answers:

1. What does this website already cover? → *content inventory*
2. Which pages may be cannibalizing each other? → *cannibalization report + matrix*
3. What does Google show? Which questions are users asking? → *SERP / PAA research*
4. What are Reddit / Quora / forums discussing? → *community research*
5. What are competitors covering that the site is not? → *competitor gap analysis*
6. What does AI search reveal? → *LLM research via official APIs*
7. Which topics are missing, which pages should be expanded, which new pages are worth creating?
8. What are the highest-opportunity fan-out queries, with real search volume, intent and a target page for each?

---

## Contents

- [Quick start – no API keys needed](#quick-start--no-api-keys-needed)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Configuration (environment variables)](#configuration-environment-variables)
- [API setup](#api-setup)
- [Running the application](#running-the-application)
- [Output files](#output-files)
- [Database & caching](#database--caching)
- [Scoring, provenance and "never fabricate"](#scoring-provenance-and-never-fabricate)
- [Rate limits](#rate-limits)
- [Responsible crawling](#responsible-crawling)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Limitations](#limitations)

---

## Quick start – no API keys needed

The tool runs fully **without any API key or account**:

```bash
pip install -r requirements.txt
python main.py
```

| Free source (no key) | What it provides |
|---|---|
| Your website (crawler) | inventory, cannibalization, coverage, URL structure, internal links |
| Google / Bing / DuckDuckGo **autocomplete** | real queries people search, expanded with question words (what/why/how/can…), prepositions (for/with/without/vs…), commercial modifiers (best/price/buy…) and A–Z |
| **Wikipedia** (MediaWiki API) | subtopics and related entities of the topic |
| **Reddit public search** | real user questions and wording (best effort – Reddit may rate-limit anonymous access) |
| **Competitor URLs you supply** (`--competitors`) | competitor headings, FAQs, questions, statistics → gap analysis |

```bash
python main.py --seed "sunscreen for oily skin" --domain example.com --country India \
               --competitors "https://competitor1.com/oily-skin-sunscreen,https://competitor2.com/guide" \
               --non-interactive
```

What changes without paid APIs:

- **`search_volume`, `competition`, `cpc_*` stay empty** – real volume only exists in Google Ads. Instead every
  query seen in autocomplete gets `demand_signal_estimated` (0–1): agreement between the three engines (60 %) and
  how high it is suggested (40 %). It is clearly labelled **ESTIMATED**, is *not* a search volume, and is used in
  the score in place of volume.
- People Also Ask, ranking URLs and automatic competitor discovery need a SERP API; question-word autocomplete
  probes cover much of the same question space, and you can name competitors yourself.
- No LLM ideas – the template patterns plus the free sources produce the candidates.

A full free run sends about 60 small requests per autocomplete engine (`AUTOCOMPLETE_DEPTH=full`, engines run in
parallel, 1 s apart per engine), a few Wikipedia requests and 3 Reddit searches – everything is cached for 14 days.
Use `--autocomplete-depth basic` for 34 per engine, `--engines google,bing` to limit engines, or
`--no-free-sources` to switch them off. These are the public suggestion endpoints browsers use; they are not
contractual APIs, so they can change or throttle – the tool then skips the engine and tells you.

## How it works

```
[1/10] Crawl website ───────── robots.txt, sitemap index, internal links, canonicals, noindex, rate limits
[2/10] Content inventory ───── topic, entities, questions answered, intent, funnel stage per page
[3/10] Cannibalization ─────── every page vs every page (topic, title, H1, content, entities, intent, type, URL)
[4/10] Google SERPs ────────── 14 query patterns via SerpApi / Serper (organic, snippets, related, autocomplete)
[5/10] People Also Ask ─────── aggregation + second-level PAA expansion
[6/10] Reddit / forums ─────── Reddit API + Quora/forum discovery through the SERP API
[7/10] Competitors ─────────── fetch top-ranking pages (robots-aware), extract headings/FAQ/entities/stats
[8/10] Fan-out generation ──── evidence → LLM (grounded) → templates; QC; semantic de-dup; coverage check
[9/10] Keyword metrics ─────── Google Ads Keyword Planner (batched, cached) – or null, never estimated
[10/10] Selection ──────────── intent, score 0-100, clustering, balanced selection, URLs, internal links, export
```

Key SEO logic:

- **Cannibalization ≠ shared keyword.** A pair is only HIGH risk when the pages have the *same search intent*
  **and** strongly overlapping topical purpose (topic/title/H1/content/entity similarity), adjusted for page type
  (e.g. a category page and a product page rarely compete) and URL structure (hub/spoke parents are softened).
- **Different keyword ≠ different intent.** De-duplication works on the *search need*: plural/punctuation/spelling
  variants, word permutations and neutral modifiers (`best sunscreen` / `best sunscreen india` /
  `best sunscreen online`) collapse, while `best sunscreen for oily skin` and
  `why does sunscreen make oily skin oily` stay separate (intent and question-frame guards).
- **Every query gets one destination**: `EXISTING_PAGE_OPTIMIZATION`, `EXISTING_PAGE_EXPANSION`, `NEW_PAGE`,
  `FAQ_SECTION`, `CATEGORY_PAGE`, `PRODUCT_PAGE`, `SERVICE_PAGE`, `COMPARISON_PAGE`, `NO_ACTION`
  or `CANNIBALIZATION_REVIEW`. If two pages already compete for a query, the tool says *review*, not *create*.
- **Coverage classes**: `EXISTING_PAGE`, `PARTIALLY_COVERED`, `CONTENT_GAP`, `CANNIBALIZATION_RISK`.
- **URL recommendations follow the site's architecture** (e.g. `/blog/…`, `/collections/…`, trailing-slash
  convention). If no matching section exists, the report says so instead of inventing one silently.
- **Honest shortfalls**: if fewer than the requested number of genuinely useful queries exist, the tool reports
  that instead of padding the list.

---

## Installation

Requires **Python 3.11+**.

```bash
git clone <your-repo-url> fan-out-query
cd fan-out-query

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# Optional but recommended:
pip install -r requirements-optional.txt   # google-ads (search volume) + sentence-transformers (semantic embeddings)

cp .env.example .env                 # then fill in the keys you have
```

`sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90 MB) on first use. Without it, a hybrid
word/char TF-IDF similarity model is used automatically (works fully offline).

---

## Configuration (environment variables)

All settings come from environment variables or a local `.env` file (never committed – see `.gitignore`).
Credentials are never hard-coded and are redacted from logs.

| Variable | Purpose | Required |
|---|---|---|
| `SERP_PROVIDER` | `serpapi` (default) or `serper` | no |
| `SERP_API_KEY` | SerpApi key | for SERP/PAA/competitor/forum research |
| `SERPER_API_KEY` | Serper.dev key (when `SERP_PROVIDER=serper`) | alternative |
| `GOOGLE_ADS_DEVELOPER_TOKEN`, `GOOGLE_ADS_CLIENT_ID`, `GOOGLE_ADS_CLIENT_SECRET`, `GOOGLE_ADS_REFRESH_TOKEN`, `GOOGLE_ADS_CUSTOMER_ID` | Keyword Planner metrics | for search volume |
| `GOOGLE_ADS_LOGIN_CUSTOMER_ID` | Manager (MCC) ID if access goes through one | if applicable |
| `GOOGLE_ADS_GEO_TARGET_ID`, `GOOGLE_ADS_LANGUAGE_ID` | override geo/language criterion IDs | no |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | OpenAI provider | no |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | Claude provider (default `claude-opus-5-5`) | no |
| `PERPLEXITY_API_KEY`, `PERPLEXITY_MODEL` | Perplexity provider (default `sonar`) | no |
| `LLM_ENABLED` | `false` disables all LLM calls | no |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | Reddit official API | no |
| `FREE_SOURCES_ENABLED`, `AUTOCOMPLETE_ENGINES`, `AUTOCOMPLETE_DEPTH`, `SUGGEST_DELAY`, `WIKIPEDIA_ENABLED`, `REDDIT_PUBLIC_ENABLED` | key-free sources (on by default) | no |
| `COMPETITOR_URLS` | comma-separated competitor pages to analyse (same as `--competitors`) | no |
| `CRAWL_DELAY`, `CRAWL_MAX_PAGES`, `CRAWL_TIMEOUT`, `CRAWL_MAX_RETRIES`, `CRAWL_USER_AGENT`, `CRAWL_RESPECT_ROBOTS` | crawler behaviour | no |
| `MAX_SERP_QUERIES`, `MAX_COMPETITOR_PAGES` | research budget | no |
| `USE_SENTENCE_TRANSFORMERS`, `EMBEDDING_MODEL` | similarity backend | no |
| `FANOUT_DB_PATH`, `FANOUT_OUTPUT_DIR`, `FANOUT_LOG_DIR` | paths | no |

**Every provider is optional.** If one is missing or fails (bad key, quota, outage), that research step is
skipped, the reason is logged and listed under *Data notes* in the summary, and the run continues.

---

## API setup

### Google Ads Keyword Planner (search volume, competition, CPC)

Search volume comes **only** from `KeywordPlanIdeaService.GenerateKeywordHistoricalMetrics`.

1. Create (or use) a **Google Ads manager account** and a client account you can access.
2. In the manager account: *Tools → API Center* → apply for a **developer token**. A token with only *Test
   account* access cannot return real metrics; you need at least **Basic access**.
3. In Google Cloud Console: create a project, **enable the Google Ads API**, then create an **OAuth client ID**
   (type *Desktop app*). Note the client ID and secret.
4. Generate a **refresh token** for the scope `https://www.googleapis.com/auth/adwords`, e.g. with the
   `generate_user_credentials.py` example shipped in the `google-ads-python` repository
   (`examples/authentication/`), or the OAuth 2.0 Playground configured with your own client ID/secret.
5. Put the values in `.env`. `GOOGLE_ADS_CUSTOMER_ID` is the 10-digit account ID (dashes are stripped
   automatically). Set `GOOGLE_ADS_LOGIN_CUSTOMER_ID` if you access the account through a manager account.
6. `pip install google-ads`

Country and language are mapped to Google Ads geo-target / language constants (e.g. India → `2356`,
English → `1000`). Unknown countries are resolved via `GeoTargetConstantService`, or set
`GOOGLE_ADS_GEO_TARGET_ID` / `GOOGLE_ADS_LANGUAGE_ID` explicitly.
`cpc_low` / `cpc_high` are Keyword Planner's **top-of-page bid** range (low/high) in the account currency.
Depending on account status Google may return rounded values – the tool reports exactly what the API returns.

### SERP provider (Google results without scraping Google)

Google is never scraped directly. Choose one:

- **SerpApi** – sign up at https://serpapi.com, copy the key from the dashboard → `SERP_API_KEY=...`
  (`SERP_PROVIDER=serpapi`). Uses `engine=google` and `engine=google_autocomplete`.
- **Serper.dev** – sign up at https://serper.dev → `SERPER_API_KEY=...` and `SERP_PROVIDER=serper`.

A default run uses ~14 SERP calls + 6 autocomplete calls + 4 PAA expansions + 4 forum discovery calls
(≈ 28 requests). All responses are cached for 14 days.

### OpenAI

Create a key at https://platform.openai.com/api-keys → `OPENAI_API_KEY=...`. Optionally set `OPENAI_MODEL`.

### Anthropic (Claude)

Create a key at https://console.anthropic.com → *API Keys* → `ANTHROPIC_API_KEY=...`.
Default model: `claude-opus-5-5` (override with `ANTHROPIC_MODEL`). Uses the official `anthropic` SDK with
server-side refusal fallbacks enabled.

### Perplexity

Create a key in Perplexity *Settings → API* → `PERPLEXITY_API_KEY=...`. Perplexity answers are web-grounded;
their citation URLs are stored with each AI-search insight.

When several LLMs are configured, all of them are used for AI-search research and the first available one
(Anthropic → OpenAI → Perplexity) generates evidence-grounded fan-out candidates. LLM output is always labelled
`LLM_GENERATED`. ChatGPT/Perplexity **web interfaces are never scraped**.

### Reddit

1. Go to https://www.reddit.com/prefs/apps → *create another app* → type **script**.
2. The client ID is shown under the app name; copy the secret.
3. Set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` and a descriptive `REDDIT_USER_AGENT`
   (e.g. `seo-fanout-research/1.0 by your_username`). Review Reddit's Data API terms for your use case.

Quora and niche forums are discovered through the SERP API (`site:quora.com …`, `… forum`), not scraped.

---

## Running the application

Interactive (asks for every input):

```bash
python main.py
```

```
Seed keyword: sunscreen for oily skin
Website: https://example.com
Country: India
Language [English]: English
Number of queries [100]: 100
```

Non-interactive:

```bash
python main.py --seed "sunscreen for oily skin" --domain example.com \
               --country India --language English --num 100 --non-interactive
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--fresh-crawl` | ignore the cached crawl and re-crawl the site |
| `--max-pages 500` | crawl budget |
| `--crawl-delay 2` | seconds between requests (robots.txt `Crawl-delay` is honoured if larger) |
| `--no-llm` | disable all LLM providers |
| `--competitors URL1,URL2` | competitor pages to analyse (no API needed) |
| `--autocomplete-depth basic\|full` | autocomplete expansion size (default `full`) |
| `--engines google,bing,duckduckgo` | autocomplete engines to use |
| `--no-free-sources` | disable autocomplete / Wikipedia / Reddit public search |
| `--no-robots` | ignore robots.txt (**only for sites you own**) |
| `--output-dir reports/` | output location |
| `-v` | show INFO logs on the console |

Example console output:

```
[1/10] Crawling website
[2/10] Building content inventory
[3/10] Detecting cannibalization
[4/10] Researching Google SERPs
[5/10] Researching PAA
[6/10] Researching Reddit/forums
[7/10] Researching competitor content
[8/10] Generating fan-out queries
[9/10] Fetching keyword metrics
[10/10] Selecting final opportunities

Website pages analyzed: 214
Potential cannibalization issues: 9 (HIGH 3, MEDIUM 6, LOW 22)
Content gaps identified: 61
Candidate queries generated: 612
Final queries: 100

Output:
  output/fanout_queries.xlsx
  output/fanout_queries.csv
  output/cannibalization_report.md
  output/content_gap_report.md
  output/research_summary.md
```

Try it locally without any API keys against the bundled fixture store:

```bash
python -m tests.fixture_site          # prints e.g. http://127.0.0.1:45123
# in another terminal:
python main.py --seed "sunscreen for oily skin" --domain http://127.0.0.1:45123 \
               --country India --num 100 --crawl-delay 0 --non-interactive
```

---

## Output files

| File | Content |
|---|---|
| `output/fanout_queries.xlsx` | 11 sheets: FINAL 100, WEBSITE INVENTORY, CANNIBALIZATION, CONTENT GAPS, TOPIC CLUSTERS, SEARCH VOLUME, SERP RESEARCH, COMMUNITY RESEARCH, COMPETITOR GAPS, QUERY SOURCES, RECOMMENDED ACTIONS |
| `output/fanout_queries.csv` | the final query list |
| `output/cannibalization_report.md` | HIGH / MEDIUM / LOW pairs with overlap, reasons, stronger page and action + similarity matrix |
| `output/content_gap_report.md` | per gap: volume, intent, existing & competitor coverage, opportunity, page type, URL, internal links |
| `output/research_summary.md` | the *SEO Fan-Out Research Report* (summary, breakdowns, provenance, data notes) |
| `logs/research.log` | detailed log (crawl progress, API calls/errors, skipped/duplicate URLs, counts) |

Final query columns (CSV and the *FINAL 100* sheet), in order:
`query_id, fan_out_query, seed_keyword, parent_topic, topic_cluster, sub_cluster, search_volume, competition,
competition_index, cpc_low, cpc_high, intent, query_type, funnel_stage, business_relevance, ai_search_relevance,
source_frequency, serp_evidence, source_google, source_paa, source_reddit, source_forums, source_competitors,
source_llm, existing_page, existing_page_url, coverage_status, cannibalization_risk, recommended_action,
recommended_content_type, recommended_url, content_gap, fanout_score, confidence, reasoning`

plus transparency columns: `data_origin_label, search_volume_source, demand_signal_estimated,
source_bing_ddg_autocomplete, source_wikipedia, user_journey_stage, origins, evidence`.

---

## Database & caching

SQLite database at `data/fanout.sqlite3` (override with `FANOUT_DB_PATH`). Tables:

`websites, pages, page_topics, keywords, queries, serp_results, paa_questions, community_questions, competitors,
cannibalization, content_gaps, query_sources, keyword_metrics, recommendations, research_runs`
plus `api_cache` (hashed SERP/LLM/Reddit responses) and `http_cache` (compressed HTML).

- Each run gets a unique ID (`YYYYMMDD-HHMMSS-xxxxxx`) stored in `research_runs` with its statistics.
- A website is crawled **once**; later runs reuse the stored pages unless `--fresh-crawl` is passed.
- SERP and Reddit responses are cached 14 days, LLM responses and keyword metrics 30 days (`cache_ttl_days`),
  crawled HTML 7 days. Cache keys are SHA-1 hashes of the request parameters.

Inspect it with any SQLite client, e.g.:

```bash
sqlite3 data/fanout.sqlite3 "SELECT id, seed_keyword, status, finished_at FROM research_runs ORDER BY started_at DESC LIMIT 5;"
```

---

## Scoring, provenance and "never fabricate"

`fanout_score` (0–100) = weighted sum of: topical relevance 20 %, intent clarity 15 %, content gap 15 %,
search volume 15 %, SERP evidence 10 %, source frequency 10 %, business relevance 10 %, AI-search relevance 5 %.
When search volume is **unavailable it is excluded and the remaining weights are re-normalised** – it is never
imputed. The component breakdown is stored per query.

Data is always labelled:

| Label | Meaning |
|---|---|
| `REAL_DATA` | crawled pages, autocomplete suggestions, SERP API results, Reddit threads, fetched competitor pages, Google Ads metrics |
| `ESTIMATED_DATA` | computed heuristics: similarity, intent, coverage, scores, `demand_signal_estimated` |
| `LLM_GENERATED` | AI-search questions and LLM fan-out ideas (hypotheses, not facts) |
| `TEMPLATE_GENERATED` | rule-based fan-out patterns and Wikipedia-subtopic queries (lowest confidence) |

If a source is unavailable the related columns are `null`/`NO` and the summary explains why. Forum content is used
for *user language and needs*, never as factual evidence. `confidence` is HIGH only with ≥ 2 independent real
sources (or a real source plus real volume).

---

## Rate limits

| Component | Behaviour |
|---|---|
| Crawler | sequential, `CRAWL_DELAY` per host (default 1 s), robots.txt `Crawl-delay` honoured (max 30 s), HTTP 429 → `Retry-After` and the crawl delay doubles for the rest of the crawl; 5xx retried with exponential back-off |
| Competitor fetches | async, max 4 concurrent, one page per competitor host, robots.txt checked first |
| SERP APIs / OpenAI / Perplexity | `tenacity` retries on 429/5xx/network errors with exponential back-off + jitter (4 attempts) |
| Anthropic | official SDK retries (4) with exponential back-off on 429/5xx/connection errors |
| Google Ads | batches of 500 keywords, `RESOURCE_EXHAUSTED` → exponential back-off (5 s, 10 s, 20 s…) |
| Reddit | API: ≥ 1 s between calls; public search: ≥ 2 s, titles/posts only |
| Autocomplete (free) | one thread per engine, `SUGGEST_DELAY` (1 s) between calls, back-off on 429/5xx, engine skipped after 5 consecutive failures |
| Wikipedia | 0.5 s between calls |

---

## Responsible crawling

- robots.txt is respected by default (`401/403` on robots.txt → treated as *disallow all*, per RFC 9309).
- Sitemaps first, then internal links; cart/checkout/account/search/admin URLs and non-HTML files are skipped.
- URL normalisation removes tracking parameters, fragments, default ports and `index.html`; `www`/non-`www` and
  trailing-slash variants are de-duplicated; canonical, `noindex` (meta + `X-Robots-Tag`) and duplicate-content
  pages are recorded but excluded from analysis.
- A clear `User-Agent` is sent (`CRAWL_USER_AGENT`) – put your contact details in it.
- `CRAWL_MAX_PAGES` caps the crawl; pages are cached so repeated runs do not hit the site again.
- The tool **never modifies** the website – it only produces recommendations.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Website unreachable` | check the domain/scheme (`http://` vs `https://`), firewall, or try `--crawl-delay 3` if the server rate-limits |
| Very few pages crawled | robots.txt may block the user agent; check `logs/research.log` for `Disallowed by robots.txt`; raise `--max-pages`; JS-rendered sites expose little HTML – make sure the XML sitemap is complete |
| Many `HTTP 403` | the site blocks bots (WAF/Cloudflare). Whitelist the user agent if it is your site |
| `search_volume` empty | Google Ads not configured, developer token lacks Basic access, wrong customer ID, or no data for the keyword – see *Data notes* in `research_summary.md` |
| `Google Ads auth error` | refresh token generated for another OAuth client, or missing `GOOGLE_ADS_LOGIN_CUSTOMER_ID` for MCC access |
| SERP columns empty | `SERP_API_KEY` / `SERPER_API_KEY` missing or quota exhausted (HTTP 401/429 in the log) |
| Fewer than 100 final queries | intended when evidence is thin; keep `AUTOCOMPLETE_DEPTH=full`, add `--competitors`, or add a SERP/LLM key to widen the candidate pool |
| `Autocomplete endpoints returned nothing` | no internet, a proxy/firewall, or the engine throttled you – wait, raise `SUGGEST_DELAY`, or use `--engines` to pick others |
| Stale results | `--fresh-crawl` for the site; delete `data/fanout.sqlite3` to clear all caches |
| `ModuleNotFoundError: google` | `pip install google-ads` |

---

## Project structure

```
main.py                        CLI
seo_fanout/
  config.py                    settings from env / .env
  logging_setup.py             logs/research.log with secret redaction
  db.py                        SQLite schema, API + HTTP caches
  http.py                      polite fetcher (retries, back-off, 429 handling, caching)
  models.py                    Pydantic models & enums
  textutil.py                  normalisation, n-grams, slugs
  crawler/  urls.py robots.py sitemap.py parser.py crawler.py
  analysis/ embeddings.py inventory.py intent.py cannibalization.py topics.py
  providers/ free_sources.py serp.py google_ads.py openai.py anthropic.py perplexity.py reddit.py llm.py base.py locale.py
  research/ free_research.py serp_research.py paa.py community.py competitors.py llm_research.py keyword_metrics.py
  queries/  generator.py dedup.py quality.py coverage.py url_recommender.py internal_links.py scoring.py selection.py
  export/   tables.py excel.py csv_export.py reports.py
  pipeline.py                  10-stage orchestration
tests/                         unit + end-to-end tests with a local fixture website
```

---

## Tests

```bash
python -m pytest
```

The suite spins up a local fixture store (with deliberate cannibalization, `noindex`, canonical duplicates, a
robots-disallowed path and a sitemap index) plus a small competitor site, and runs the full pipeline – with no
API keys, and with test-only fake SERP / LLM / Keyword-Planner providers to exercise every code path.
The fakes live only in `tests/`; the application itself never fabricates data.

---

## Limitations

- Templates and intent rules are English-centric; other languages work best with an LLM provider configured.
- Without `sentence-transformers`, similarity is lexical (TF-IDF + light synonym canonicalisation), so
  de-duplication is deliberately conservative.
- JavaScript-only content is not rendered (no headless browser); rely on sitemaps for discovery.
- Cannibalization is inferred from on-page signals; confirm HIGH-risk pairs with Google Search Console
  query-to-page data before merging or redirecting.
