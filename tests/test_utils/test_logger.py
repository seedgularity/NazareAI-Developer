# tests/test_utils/test_logger.py

import pytest
import logging
import os
from pathlib import Path

from src.utils.logger import get_logger, log_exception


@pytest.fixture
def temp_log_dir(tmp_path):
    """Fixture to provide temporary log directory"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


@pytest.fixture
def temp_log_file(temp_log_dir):
    """Fixture to provide temporary log file"""
    return temp_log_dir / "test.log"


def test_get_logger_basic():
    """Test basic logger creation"""
    logger = get_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_get_logger_with_file(temp_log_file):
    """Test logger with file output"""
    logger = get_logger("test_file_logger", log_file=temp_log_file)

    # Verify logger configuration
    assert len(logger.handlers) == 2  # Console and file handlers
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    # Test logging to file
    test_message = "Test log message"
    logger.info(test_message)

    # Verify log file content
    assert temp_log_file.exists()
    log_content = temp_log_file.read_text()
    assert test_message in log_content


def test_get_logger_custom_level():
    """Test logger with custom log level"""
    logger = get_logger("test_debug_logger", log_level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_log_exception(temp_log_file):
    """Test exception logging"""
    logger = get_logger("test_exception_logger", log_file=temp_log_file)
    test_error = ValueError("Test error")

    try:
        raise test_error
    except Exception as e:
        log_exception(logger, e)

    # Verify exception was logged
    log_content = temp_log_file.read_text()
    assert "ValueError" in log_content
    assert "Test error" in log_content


def test_multiple_loggers_same_name():
    """Test logger instance reuse"""
    logger1 = get_logger("test_multiple")
    logger2 = get_logger("test_multiple")

    # Verify same logger instance is returned
    assert logger1 is logger2
    assert id(logger1) == id(logger2)


def test_logger_handler_duplication():
    """Test prevention of handler duplication"""
    logger = get_logger("test_duplication")
    initial_handler_count = len(logger.handlers)

    # Get same logger again
    _ = get_logger("test_duplication")

    # Verify no handlers were added
    assert len(logger.handlers) == initial_handler_count


def test_logger_directory_creation(temp_log_dir):
    """Test log directory creation"""
    nested_log_file = temp_log_dir / "nested" / "deep" / "test.log"

    logger = get_logger("test_nested_logger", log_file=nested_log_file)
    logger.info("Test message")

    # Verify directory structure was created
    assert nested_log_file.exists()
    assert nested_log_file.parent.is_dir()