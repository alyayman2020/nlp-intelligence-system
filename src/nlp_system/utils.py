"""Shared utilities: logging, seeding, timing."""

from __future__ import annotations

import logging
import random
import sys
import time
from contextlib import contextmanager

import numpy as np

_CONFIGURED = False


def get_logger(name: str = "nlp_system", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """Context manager that logs the wall-clock duration of a block."""
    log = logger or get_logger()
    start = time.perf_counter()
    log.info("▶ %s …", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("✓ %s done in %.2fs", label, elapsed)
