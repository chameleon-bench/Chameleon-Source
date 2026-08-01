"""
Unified logging configuration module for the project.

Provides standardized log format, including:
- Date and time
- File path (relative path)
- Line number
- Log level
- Log message

Usage example:
    from utils.logging_config import setup_logger, get_logger

    # Set up logging (call once)
    setup_logger()

    # Get logger
    logger = get_logger(__name__)
    logger.info("This is an info log")
    logger.error("This is an error log")
"""

import logging
import sys
import io
from pathlib import Path
from datetime import datetime


def _get_utf8_stdout():
    """
    Get the stream for log output.

    Strategy: return sys.stdout directly without changing its encoding.
    Console output follows the system default encoding (Windows=GBK, Mac/Linux=UTF-8),
    file output uses UTF-8 uniformly (specified in FileHandler).
    This avoids cross-platform encoding issues.
    """
    return sys.stdout


# Log root directory
PROJECT_ROOT = Path(__file__).parent.parent


class RelativePathFormatter(logging.Formatter):
    """Custom formatter that displays relative paths."""

    def format(self, record):
        # Get absolute path
        pathname = record.pathname

        # Convert to relative path
        try:
            relative_path = Path(pathname).relative_to(PROJECT_ROOT)
            record.pathname_rel = str(relative_path).replace('\\', '/')
        except (ValueError, TypeError):
            # If conversion fails, use the original path
            record.pathname_rel = pathname

        return super().format(record)


def setup_logger(
    name: str = "translation_benchmark",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: str = None
):
    """
    Set up the project logging system.

    Args:
        name: Logger name
        level: Log level (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to output to console
        log_dir: Log file directory (default: logs/)
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Log format: time | relative_path:line | level | message
    log_format = "%(asctime)s | %(pathname_rel)s:%(lineno)d | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = RelativePathFormatter(log_format, date_format)

    # Console handler - use UTF-8 encoding, compatible with Windows/Mac/Linux
    if log_to_console:
        console_stream = _get_utf8_stdout()
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = PROJECT_ROOT / "logs"

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file by date
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{name}_{today}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name, typically use __name__

    Returns:
        Logger instance

    Usage example:
        from utils.logging_config import get_logger

        logger = get_logger(__name__)
        logger.info("Log message")
    """
    if name is None:
        name = "translation_benchmark"

    logger = logging.getLogger(name)

    # If logger has no handlers yet, use default config
    if not logger.handlers:
        setup_logger(name=name)

    return logger


def create_child_logger(parent_name: str, child_name: str) -> logging.Logger:
    """
    Create a child logger.

    Args:
        parent_name: Parent logger name
        child_name: Child logger name

    Returns:
        Child logger instance

    Usage example:
        from utils.logging_config import create_child_logger

        logger = create_child_logger("translation_benchmark", "analyzer")
    """
    full_name = f"{parent_name}.{child_name}" if parent_name else child_name
    return logging.getLogger(full_name)


# Initialize default logger
_default_logger = None


def init_project_logging(level: int = logging.INFO):
    """
    Initialize project logging (call once at program entry).

    Args:
        level: Log level
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger(level=level)


def log_function_call(logger: logging.Logger):
    """
    Decorator: log function calls.

    Args:
        logger: Logger instance

    Usage example:
        @log_function_call(get_logger(__name__))
        def my_function(arg1, arg2):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling function: {func.__name__} args={args} kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function completed: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Function exception: {func.__name__} error={str(e)}")
                raise
        return wrapper
    return decorator
