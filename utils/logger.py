"""
InsightAI - Logging helper.

Provides a single, configurable logger used across the application. The log
level is driven by the ``LOG_LEVEL`` setting (default INFO).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "insightai") -> logging.Logger:
    """Return a project logger, configured once per process."""
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        try:
            from config.settings import get_settings

            level = get_settings().log_level
        except Exception:
            level = "INFO"
        level = getattr(logging, level, logging.INFO)

        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    return logger
