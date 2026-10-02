#!/usr/bin/env python3
"""SEO Fan-Out Query Research & Cannibalization Analysis – command-line entry point.

Interactive:      python main.py
Non-interactive:  python main.py --seed "sunscreen for oily skin" --domain example.com --country India \
                      --language English --num 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from seo_fanout import __version__
from seo_fanout.config import load_settings
from seo_fanout.crawler.crawler import WebsiteUnavailableError
from seo_fanout.crawler.urls import InvalidDomainError, validate_domain
from seo_fanout.logging_setup import setup_logging
from seo_fanout.models import Risk
from seo_fanout.pipeline import TOTAL_STEPS, ResearchInput, ResearchPipeline


def _ask(prompt: str, default: str | None = None, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            value = ""
        if not value and default is not None:
            return default
        if value or not required:
            return value
        print("  This field is required.")


def _ask_domain(initial: str | None) -> str:
    value = initial
    while True:
        if not value:
            value = _ask("Website", required=True)
        try:
            return validate_domain(value)
        except InvalidDomainError as exc:
            print(f"  {exc} Please enter a domain such as example.com or https://example.com")
            value = None


def _ask_int(prompt: str, default: int, initial: int | None) -> int:
    if initial is not None:
        return initial
    while True:
        raw = _ask(prompt, str(default))
        try:
            n = int(raw)
            if 1 <= n <= 1000:
                return n
        except ValueError:
            pass
        print("  Enter a whole number between 1 and 1000.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cannibalization-aware SEO/GEO fan-out query research")
    p.add_argument("--seed", help="Seed keyword / topic")
    p.add_argument("--domain", help="Website domain, e.g. example.com")
    p.add_argument("--country", help="Target country (e.g. India)")
    p.add_argument("--language", help="Target language (e.g. English)")
    p.add_argument("--num", type=int, help="Number of fan-out queries (default 100)")
    p.add_argument("--fresh-crawl", action="store_true", help="Ignore the cached crawl and re-crawl the website")
    p.add_argument("--max-pages", type=int, help="Maximum pages to crawl")
    p.add_argument("--crawl-delay", type=float, help="Seconds between requests to the website (min, robots.txt may raise it)")
    p.add_argument("--no-robots", action="store_true", help="Do not apply robots.txt rules (only for sites you own)")
    p.add_argument("--no-llm", action="store_true", help="Disable all LLM providers")
    p.add_argument("--output-dir", help="Directory for output files (default ./output)")
    p.add_argument("--non-interactive", action="store_true", help="Fail instead of prompting for missing input")
    p.add_argument("-v", "--verbose", action="store_true", help="Show INFO logs on the console")
    p.add_argument("--version", action="version", version=f"seo-fanout {__version__}")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    if args.max_pages:
        settings.crawl.max_pages = args.max_pages
    if args.crawl_delay is not None:
        settings.crawl.crawl_delay = max(0.0, args.crawl_delay)
    if args.no_robots:
        settings.crawl.respect_robots = False
    if args.no_llm:
        settings.llm_enabled = False
    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
    log = setup_logging(settings.log_dir, args.verbose)

    print("=" * 64)
    print(" SEO Fan-Out Query Research & Cannibalization Analysis")
    print("=" * 64)
    if args.non_interactive:
        if not args.seed or not args.domain:
            print("--seed and --domain are required in --non-interactive mode", file=sys.stderr)
            return 2
        try:
            base_url = validate_domain(args.domain)
        except InvalidDomainError as exc:
            print(f"Invalid domain: {exc}", file=sys.stderr)
            return 2
        seed, country, language, num = args.seed, args.country or "", args.language or "English", args.num or 100
    else:
        seed = args.seed or _ask("Seed keyword", required=True)
        base_url = _ask_domain(args.domain)
        country = args.country if args.country is not None else _ask("Country", "")
        language = args.language or _ask("Language", "English")
        num = _ask_int("Number of queries", 100, args.num)

    print("\nProviders:")
    for name, ok in settings.provider_status().items():
        print(f"  {'✓' if ok else '–'} {name}{'' if ok else ' (not configured – skipped)'}")
    print()

    started = time.time()

    def progress(step: int, label: str) -> None:
        print(f"[{step}/{TOTAL_STEPS}] {label}", flush=True)

    pipeline = ResearchPipeline(settings, progress)
    try:
        result, paths = pipeline.run(
            ResearchInput(seed=seed, base_url=base_url, country=country, language=language,
                          num_queries=num, fresh_crawl=args.fresh_crawl)
        )
    except WebsiteUnavailableError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        log.error("Website unavailable: %s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        log.exception("Run failed")
        print(f"\nERROR: {type(exc).__name__}: {exc}\nSee logs/research.log for details.", file=sys.stderr)
        return 1
    finally:
        pipeline.close()

    risks = {r: sum(1 for p in result.pairs if p.risk == r) for r in (Risk.HIGH, Risk.MEDIUM, Risk.LOW)}
    print()
    print(f"Website pages analyzed: {len(result.inventory)}" + (" (cached crawl)" if result.crawl_reused else ""))
    print(f"Potential cannibalization issues: {risks[Risk.HIGH] + risks[Risk.MEDIUM]} "
          f"(HIGH {risks[Risk.HIGH]}, MEDIUM {risks[Risk.MEDIUM]}, LOW {risks[Risk.LOW]})")
    print(f"Content gaps identified: {result.stats.get('content_gaps_identified', 0)}")
    print(f"Candidate queries generated: {result.stats.get('candidates_generated', 0)}")
    print(f"Final queries: {len(result.final)}")
    shortfall = result.stats.get("selection", {}).get("shortfall_reason")
    if shortfall:
        print(f"  NOTE: {shortfall}")
    if result.data_notes:
        print("\nData notes:")
        for note in result.data_notes[:10]:
            print(f"  - {note}")
    print("\nOutput:")
    for path in paths.values():
        try:
            print(f"  {Path(path).resolve().relative_to(Path.cwd())}")
        except ValueError:
            print(f"  {path}")
    print(f"\nRun ID: {result.run_id} · {time.time() - started:.1f}s · log: {settings.log_dir / 'research.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
