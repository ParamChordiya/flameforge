"""Structured file logging for FlameForge.

The TUI surfaces only summaries to keep the screen clean, but everything is
written to a rotating log file (``flameforge.log`` by default) so users have a
full trace for debugging and bug reports. Logging never raises into the caller:
if the log file cannot be opened we fall back to a null handler.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flameforge.constants import APP_NAME, LOG_FILENAME

_LOGGER_NAME = "flameforge"
_configured = False


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root FlameForge logger.

    Safe to call multiple times; configuration happens only once. If the log file
    cannot be created, a null handler is installed so logging calls are no-ops
    rather than errors.

    Args:
        log_dir: Directory for the log file; defaults to the current directory.
        level: The logging level for the file handler.

    Returns:
        The configured ``flameforge`` logger.
    """
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_path = Path(log_dir) / LOG_FILENAME if log_dir else Path(LOG_FILENAME)
    try:
        handler: logging.Handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    except OSError:
        handler = logging.NullHandler()

    logger.addHandler(handler)
    logger.info("%s logging initialized (file=%s)", APP_NAME, log_path)
    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``flameforge`` namespace.

    Args:
        name: Optional dotted suffix (e.g. "training"); None returns the root.

    Returns:
        A configured logger. Calls :func:`setup_logging` if needed.
    """
    if not _configured:
        setup_logging()
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
