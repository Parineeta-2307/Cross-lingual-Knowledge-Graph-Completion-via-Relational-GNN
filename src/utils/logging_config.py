"""Structured logging configuration using loguru.

Why loguru over stdlib logging:
    loguru provides structured, colored console output and automatic file rotation
    with zero boilerplate. In a research pipeline where we need to trace SPARQL
    failures, cache behavior, and Unicode edge cases, readable logs are critical.
    A single setup_logging() call configures the entire project.

Usage:
    from src.utils.logging_config import setup_logging
    from loguru import logger

    setup_logging()  # Call once at program start
    logger.info("Pipeline started | phase=1")
"""

import sys
import os
from pathlib import Path
from loguru import logger


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru for the entire project.

    Sets up two output sinks:
      1. Console (stderr) — colorized, human-readable, INFO+ by default
      2. File — structured with timestamps, rotated, captures DEBUG level
         for post-mortem analysis of pipeline issues

    Args:
        log_dir: Directory for log files (created if missing).
        log_level: Minimum log level for console output.
            One of: DEBUG, INFO, WARNING, ERROR.
        rotation: When to rotate log files (e.g., "10 MB", "1 day").
        retention: How long to keep old rotated files (e.g., "7 days").

    Returns:
        None

    Example:
        >>> setup_logging(log_level="DEBUG")
        >>> from loguru import logger
        >>> logger.info("Cache hit | query_hash=abc123")
    """
    # Remove default loguru handler to prevent duplicate console output.
    # loguru ships with one default stderr handler — we replace it with ours.
    logger.remove()

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # --- Console handler: colorized, concise ---
    # This is what you see in your terminal during development.
    # Shows timestamp, level, module name, and message.
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Allow environment variable override for log level
    effective_level = os.environ.get("LOG_LEVEL", log_level)

    logger.add(
        sys.stderr,
        format=console_format,
        level=effective_level,
        colorize=True,
    )

    # --- File handler: detailed, with rotation ---
    # This captures everything including DEBUG messages.
    # Useful for debugging SPARQL failures or Unicode issues after the fact.
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    logger.add(
        str(log_path / "pipeline.log"),
        format=file_format,
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",  # Required for Japanese/German/Dutch log messages
    )

    logger.info(
        f"Logging initialized | level={effective_level} | log_dir={log_dir}"
    )
