"""Structured, human-readable pipeline logging."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

_CONFIGURED = False


def get_logger(name: str = "minedesign") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
        root = logging.getLogger("minedesign")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)


@contextmanager
def step(label: str, logger: logging.Logger | None = None):
    """Time and announce one pipeline stage."""
    log = logger or get_logger()
    log.info("-> %s", label)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("   %s done in %.2fs", label, time.perf_counter() - t0)
