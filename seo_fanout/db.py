"""SQLite persistence and caching layer.

All research artefacts are stored so a run can be resumed or re-analysed
without re-crawling or re-paying for API calls.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_setup import get_logger
from .textutil import sha1

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS websites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT UNIQUE NOT NULL,
    base_url TEXT NOT NULL,
    robots_txt TEXT,
    sitemaps TEXT,
    last_crawled_at TEXT,
    pages_crawled INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    website_id INTEGER NOT NULL REFERENCES websites(id),
    url TEXT NOT NULL,
    url_hash TEXT NOT NULL,
    status INTEGER,
    canonical TEXT,
    indexable INTEGER,
    page_type TEXT,
    title TEXT,
    h1 TEXT,
    word_count INTEGER,
    content_hash TEXT,
    data TEXT NOT NULL,
    crawled_at TEXT,
    UNIQUE(website_id, url_hash)
);
CREATE TABLE IF NOT EXISTS page_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    website_id INTEGER NOT NULL,
    url TEXT NOT NULL,
    primary_topic TEXT,
    search_intent TEXT,
    funnel_stage TEXT,
    data TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(website_id, url)
);
CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    website_id INTEGER,
    seed_keyword TEXT NOT NULL,
    country TEXT,
    language TEXT,
    num_queries INTEGER,
    status TEXT,
    stats TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    keyword TEXT,
    role TEXT,
    UNIQUE(run_id, keyword, role)
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    query TEXT NOT NULL,
    normalized TEXT,
    selected INTEGER DEFAULT 0,
    rejected_reason TEXT,
    score REAL,
    data TEXT,
    UNIQUE(run_id, normalized)
);
CREATE TABLE IF NOT EXISTS query_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    query TEXT NOT NULL,
    source TEXT NOT NULL,
    evidence TEXT,
    label TEXT
);
CREATE TABLE IF NOT EXISTS serp_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    query TEXT NOT NULL,
    provider TEXT,
    position INTEGER,
    url TEXT,
    title TEXT,
    snippet TEXT,
    result_type TEXT
);
CREATE TABLE IF NOT EXISTS paa_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    source_query TEXT,
    question TEXT,
    snippet TEXT,
    link TEXT
);
CREATE TABLE IF NOT EXISTS community_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    question TEXT,
    category TEXT,
    frequency INTEGER,
    sources TEXT,
    urls TEXT,
    confidence TEXT
);
CREATE TABLE IF NOT EXISTS competitors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    url TEXT,
    domain TEXT,
    serp_position INTEGER,
    data TEXT
);
CREATE TABLE IF NOT EXISTS cannibalization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    website_id INTEGER,
    url_a TEXT,
    url_b TEXT,
    risk TEXT,
    combined_score REAL,
    data TEXT
);
CREATE TABLE IF NOT EXISTS content_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    query TEXT,
    gap_type TEXT,
    data TEXT
);
CREATE TABLE IF NOT EXISTS keyword_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword_norm TEXT NOT NULL,
    country TEXT NOT NULL,
    language TEXT NOT NULL,
    source TEXT,
    search_volume INTEGER,
    data TEXT,
    fetched_at REAL,
    UNIQUE(keyword_norm, country, language, source)
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    query TEXT,
    action TEXT,
    target_url TEXT,
    data TEXT
);
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key TEXT PRIMARY KEY,
    namespace TEXT,
    payload BLOB,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS http_cache (
    url_hash TEXT PRIMARY KEY,
    url TEXT,
    status INTEGER,
    headers TEXT,
    body BLOB,
    fetched_at REAL
);
CREATE INDEX IF NOT EXISTS idx_pages_site ON pages(website_id);
CREATE INDEX IF NOT EXISTS idx_queries_run ON queries(run_id);
CREATE INDEX IF NOT EXISTS idx_sources_run ON query_sources(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ helpers
    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    # ----------------------------------------------------------------- websites
    def get_or_create_website(self, domain: str, base_url: str) -> int:
        rows = self.query("SELECT id FROM websites WHERE domain=?", (domain,))
        if rows:
            return int(rows[0]["id"])
        cur = self.execute("INSERT INTO websites(domain, base_url) VALUES(?,?)", (domain, base_url))
        return int(cur.lastrowid)

    def website_info(self, website_id: int) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM websites WHERE id=?", (website_id,))
        return rows[0] if rows else None

    def mark_crawled(self, website_id: int, pages: int, robots_txt: str | None, sitemaps: list[str]) -> None:
        self.execute(
            "UPDATE websites SET last_crawled_at=?, pages_crawled=?, robots_txt=?, sitemaps=? WHERE id=?",
            (_now(), pages, robots_txt, json.dumps(sitemaps), website_id),
        )

    def clear_pages(self, website_id: int) -> None:
        self.execute("DELETE FROM pages WHERE website_id=?", (website_id,))
        self.execute("DELETE FROM page_topics WHERE website_id=?", (website_id,))

    # -------------------------------------------------------------------- pages
    def save_page(self, website_id: int, page: Any) -> None:
        data = page.model_dump_json()
        self.execute(
            """INSERT INTO pages(website_id,url,url_hash,status,canonical,indexable,page_type,title,h1,
                   word_count,content_hash,data,crawled_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(website_id,url_hash) DO UPDATE SET status=excluded.status,
                   canonical=excluded.canonical, indexable=excluded.indexable,
                   page_type=excluded.page_type, title=excluded.title, h1=excluded.h1,
                   word_count=excluded.word_count, content_hash=excluded.content_hash,
                   data=excluded.data, crawled_at=excluded.crawled_at""",
            (
                website_id, page.url, sha1(page.url), page.status, page.canonical,
                int(page.indexable), page.page_type.value, page.title,
                page.h1[0] if page.h1 else "", page.word_count, page.content_hash, data, _now(),
            ),
        )

    def load_pages(self, website_id: int) -> list[dict]:
        rows = self.query("SELECT data FROM pages WHERE website_id=? ORDER BY id", (website_id,))
        return [json.loads(r["data"]) for r in rows]

    def save_page_topics(self, website_id: int, inventory: list[Any]) -> None:
        self.executemany(
            """INSERT INTO page_topics(website_id,url,primary_topic,search_intent,funnel_stage,data,updated_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(website_id,url) DO UPDATE SET primary_topic=excluded.primary_topic,
                 search_intent=excluded.search_intent, funnel_stage=excluded.funnel_stage,
                 data=excluded.data, updated_at=excluded.updated_at""",
            [
                (website_id, inv.url, inv.primary_topic, inv.search_intent.value,
                 inv.funnel_stage, inv.model_dump_json(), _now())
                for inv in inventory
            ],
        )

    # --------------------------------------------------------------------- runs
    def create_run(self, run_id: str, website_id: int, seed: str, country: str, language: str, n: int) -> None:
        self.execute(
            """INSERT INTO research_runs(id,website_id,seed_keyword,country,language,num_queries,status,started_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (run_id, website_id, seed, country, language, n, "running", _now()),
        )

    def finish_run(self, run_id: str, status: str, stats: dict) -> None:
        self.execute(
            "UPDATE research_runs SET status=?, stats=?, finished_at=? WHERE id=?",
            (status, json.dumps(stats, default=str), _now(), run_id),
        )

    # -------------------------------------------------------------- api caching
    def cache_get(self, namespace: str, key_material: str, ttl_days: float | None = 30) -> Any | None:
        key = sha1(f"{namespace}|{key_material}")
        rows = self.query("SELECT payload, created_at FROM api_cache WHERE cache_key=?", (key,))
        if not rows:
            return None
        if ttl_days is not None and time.time() - rows[0]["created_at"] > ttl_days * 86400:
            return None
        try:
            return json.loads(zlib.decompress(rows[0]["payload"]).decode("utf-8"))
        except Exception:  # corrupted cache entry -> treat as miss
            return None

    def cache_set(self, namespace: str, key_material: str, value: Any) -> None:
        key = sha1(f"{namespace}|{key_material}")
        payload = zlib.compress(json.dumps(value, default=str).encode("utf-8"))
        self.execute(
            "INSERT OR REPLACE INTO api_cache(cache_key, namespace, payload, created_at) VALUES(?,?,?,?)",
            (key, namespace, payload, time.time()),
        )

    # ------------------------------------------------------------- http caching
    def http_cache_get(self, url: str, ttl_days: float = 7) -> dict | None:
        rows = self.query("SELECT * FROM http_cache WHERE url_hash=?", (sha1(url),))
        if not rows or time.time() - rows[0]["fetched_at"] > ttl_days * 86400:
            return None
        row = rows[0]
        return {
            "url": row["url"],
            "status": row["status"],
            "headers": json.loads(row["headers"] or "{}"),
            "text": zlib.decompress(row["body"]).decode("utf-8", errors="replace") if row["body"] else "",
        }

    def http_cache_set(self, url: str, status: int, headers: dict, text: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO http_cache(url_hash,url,status,headers,body,fetched_at) VALUES(?,?,?,?,?,?)",
            (sha1(url), url, status, json.dumps(headers), zlib.compress(text.encode("utf-8")), time.time()),
        )

    # ---------------------------------------------------------- keyword metrics
    def get_keyword_metrics(self, keyword_norm: str, country: str, language: str, source: str, ttl_days: float) -> dict | None:
        rows = self.query(
            "SELECT data, fetched_at FROM keyword_metrics WHERE keyword_norm=? AND country=? AND language=? AND source=?",
            (keyword_norm, country, language, source),
        )
        if not rows or time.time() - rows[0]["fetched_at"] > ttl_days * 86400:
            return None
        return json.loads(rows[0]["data"])

    def save_keyword_metrics(self, keyword_norm: str, country: str, language: str, source: str, metrics: dict) -> None:
        self.execute(
            """INSERT OR REPLACE INTO keyword_metrics(keyword_norm,country,language,source,search_volume,data,fetched_at)
               VALUES(?,?,?,?,?,?,?)""",
            (keyword_norm, country, language, source, metrics.get("search_volume"),
             json.dumps(metrics, default=str), time.time()),
        )
