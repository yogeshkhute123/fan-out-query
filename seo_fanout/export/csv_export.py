"""CSV exporter for the final query list."""

from __future__ import annotations

from pathlib import Path

from ..results import ResearchResult
from .tables import final_df


def write_csv(result: ResearchResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_df(result).to_csv(path, index=False, encoding="utf-8-sig")
    return path
