"""Centralized logging configuration for wordcel."""

import logging
import os
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Centralized logging configuration for wordcel.

    This function sets the root logger level to WARNING to silence external libraries
    and then sets the 'wordcel' logger to the specified or default level.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to WORDCEL_LOG_LEVEL env var or INFO
        format_string: Custom format string. Defaults to WORDCEL_LOG_FORMAT env var or standard format
        log_file: Optional log file path. Uses WORDCEL_LOG_FILE env var if provided
    """
    level = level or os.getenv("WORDCEL_LOG_LEVEL", "INFO")
    format_string = format_string or os.getenv(
        "WORDCEL_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = log_file or os.getenv("WORDCEL_LOG_FILE")

    # Configure handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)

    # Configure root logger to be less verbose
    logging.basicConfig(
        level=logging.WARNING,
        format=format_string,
        handlers=handlers,
        force=True,  # Override existing config
    )

    # Set the desired level for the wordcel package
    wordcel_logger = logging.getLogger("wordcel")
    wordcel_logger.setLevel(getattr(logging, level.upper()))


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent wordcel naming.

    Args:
        name: Module name (e.g., 'dag.nodes', 'cli', 'rag')

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"wordcel.{name}")
