"""Logging configuration with secret redaction."""

from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import SECRET_ENV_VARS

_KEY_PATTERNS = [
    re.compile(r"(api[_-]?key=)[^&\s]+", re.I),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.I),
    re.compile(r"(x-api-key['\"]?:\s*['\"]?)[^'\"\s,]+", re.I),
    re.compile(r"(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]+"),
]


class SecretRedactingFilter(logging.Filter):
    """Removes any configured secret value from log records."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets = [v for k in SECRET_ENV_VARS if (v := os.getenv(k)) and len(v) >= 6]

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***REDACTED***")
        for pattern in _KEY_PATTERNS:
            text = pattern.sub(r"\1***REDACTED***", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        record.msg = self.redact(message)
        record.args = ()
        return True


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("seo_fanout")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    redactor = SecretRedactingFilter()
    file_handler = RotatingFileHandler(
        log_dir / "research.log", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    file_handler.addFilter(redactor)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter("    %(levelname)s: %(message)s"))
    console.addFilter(redactor)
    logger.addHandler(console)

    # Third-party libraries are noisy at DEBUG level.
    for noisy in ("httpx", "httpcore", "urllib3", "trafilatura", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"seo_fanout.{name}")
