"""
logger.py — Structured, timed logger for the eval generation pipeline.

Uses Python's built-in logging module (no external deps).
Outputs to both console (with colour) and to logs/eval_generation.log.
Tracks pipeline-level statistics that are printed as a summary at exit.
"""

import logging
import sys
import time
from pathlib import Path


# Ensure log directory exists
Path("logs").mkdir(exist_ok=True)

_FMT  = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "eval") -> logging.Logger:
    """Return a configured logger that writes to console + log file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT, _DATE))
    # Force UTF-8 output on Windows consoles that default to cp1252
    if hasattr(ch.stream, "reconfigure"):
        try:
            ch.stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logger.addHandler(ch)

    # File handler — DEBUG and above (full detail)
    fh = logging.FileHandler("logs/eval_generation.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(fh)

    return logger


class PipelineStats:
    """Accumulates pipeline-wide counters and prints a summary at the end."""

    def __init__(self) -> None:
        self._start            = time.time()
        self.chunks_scanned    = 0
        self.chunks_processed  = 0
        self.samples_generated = 0
        self.chunks_skipped    = 0
        self.errors            = 0

    def elapsed(self) -> float:
        return round(time.time() - self._start, 2)

    def summary(self, logger: logging.Logger) -> None:
        """Log the final statistics block using ASCII-safe separators."""
        sep = "-" * 50
        logger.info(sep)
        logger.info("PIPELINE SUMMARY")
        logger.info(sep)
        logger.info(f"  Chunks scanned    : {self.chunks_scanned}")
        logger.info(f"  Chunks processed  : {self.chunks_processed}")
        logger.info(f"  Samples generated : {self.samples_generated}")
        logger.info(f"  Chunks skipped    : {self.chunks_skipped}")
        logger.info(f"  Errors            : {self.errors}")
        logger.info(f"  Execution time    : {self.elapsed()}s")
        logger.info(sep)
