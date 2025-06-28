"""Centralized logging configuration for wordcel."""

import logging
import os
from typing import Optional


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    configure_root: bool = True,
    silence_external: bool = True,
) -> None:
    """Centralized logging configuration for wordcel.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to WORDCEL_LOG_LEVEL env var or INFO
        format_string: Custom format string. Defaults to WORDCEL_LOG_FORMAT env var or standard format
        log_file: Optional log file path. Uses WORDCEL_LOG_FILE env var if provided
        configure_root: Whether to configure the root logger
        silence_external: Whether to silence noisy external libraries
    """
    level = level or os.getenv("WORDCEL_LOG_LEVEL", "INFO")
    format_string = format_string or os.getenv(
        "WORDCEL_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = log_file or os.getenv("WORDCEL_LOG_FILE")

    if configure_root:
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

        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_string,
            handlers=handlers,
            force=True,  # Override existing config
        )

        # Silence noisy external libraries
        if silence_external:
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent wordcel naming.

    Args:
        name: Module name (e.g., 'dag.nodes', 'cli', 'rag')

    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"wordcel.{name}")
