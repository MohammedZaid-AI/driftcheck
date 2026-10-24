"""Logging setup shared by the CLI, the apps and the eval."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", force=True)
    # Third party libraries are noisy at INFO.
    for name in ("httpx", "httpcore", "urllib3", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
