# src/utils/logger.py

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(debug: bool = False):
    """
    Configure global logging settings for the application.

    Args:
        debug: If True, sets logging level to DEBUG, otherwise INFO
    """
    # Remove default logger
    logger.remove()

    # Determine log level
    log_level = "DEBUG" if debug else "INFO"

    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Add file handler for logs
    log_dir = Path.home() / ".ai_developer_worker" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ai_developer_worker.log"

    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="10 MB",
        retention="1 week"
    )


def get_logger(
        name: str,
        log_level: int = logging.INFO,
        log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Creates and returns a logger instance with consistent formatting and handlers.

    Args:
        name: The name of the logger (typically __name__)
        log_level: The logging level (default: logging.INFO)
        log_file: Optional path to a log file.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if log_file is provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance for quick access
default_logger = get_logger('ai_developer_worker')


def log_exception(logger: logging.Logger, exc: Exception, message: str = "An error occurred:"):
    """
    Helper function to consistently log exceptions across the application.

    Args:
        logger: The logger instance to use
        exc: The exception to log
        message: Optional custom message to precede the exception details
    """
    logger.error(f"{message} {str(exc)}", exc_info=True)