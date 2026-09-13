"""Structured logging for QGFDA experiments.

Every experiment run gets a plain-text log (human-readable, via the stdlib
logging module) under experiments/logs/, plus the caller is responsible for
writing the structured JSON record via experiment_tracker.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.settings import REPO_ROOT

LOG_DIR = REPO_ROOT / "experiments" / "logs"


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """Return a configured logger that writes to stdout and, optionally, a
    per-module file under experiments/logs/<name>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(".", "_")
        file_handler = logging.FileHandler(LOG_DIR / f"{safe_name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
