# src/utils/logger.py
# Logging configuration for the project.

import logging
import sys
from typing import Optional

# --- Configuration ---
# These could potentially be moved to or overridden by src/config.py
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FILE = "analysis.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# --- End Configuration ---

# Store configured loggers to avoid duplicate handlers
_configured_loggers = {}


def setup_logger(name: str = 'granger_analysis',
                 log_level: Optional[str] = None,
                 log_file: Optional[str] = None,
                 use_console: bool = True,
                 use_file: bool = True) -> logging.Logger:
    """
    Sets up and returns a logger instance.

    Args:
        name: Name of the logger.
        log_level: Logging level (e.g., 'DEBUG', 'INFO'). Overrides default/config.
        log_file: Path to the log file. Overrides default/config.
        use_console: Whether to log to the console.
        use_file: Whether to log to a file.

    Returns:
        Configured logger instance.
    """
    global _configured_loggers

    if name in _configured_loggers:
        return _configured_loggers[name]

    # Determine final log level and file path
    final_log_level_str = log_level or DEFAULT_LOG_LEVEL
    final_log_file = log_file or DEFAULT_LOG_FILE

    # Get numeric log level
    numeric_level = getattr(logging, final_log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        print(
            f"Warning: Invalid log level '{final_log_level_str}'. Defaulting to INFO.")
        numeric_level = logging.INFO
        final_log_level_str = "INFO"  # Update string representation

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Remove existing handlers to prevent duplication if function is called again
    # (Though _configured_loggers check should prevent this)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if use_file and final_log_file:
        try:
            file_handler = logging.FileHandler(
                final_log_file, mode='a')  # Append mode
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file handler for {final_log_file}: {e}")
            logger.error(f"Could not attach file handler to {final_log_file}")

    # Prevent propagation to root logger if handlers are added
    logger.propagate = False

    logger.info(
        f"Logger '{name}' configured. Level: {final_log_level_str}. File: {final_log_file if use_file else 'None'}. Console: {use_console}.")

    _configured_loggers[name] = logger
    return logger


if __name__ == '__main__':
    # Example usage
    print("Testing logger setup...")

    # Get default logger
    logger1 = setup_logger()
    logger1.debug("This is a debug message (should not appear by default).")
    logger1.info("This is an info message.")
    logger1.warning("This is a warning message.")
    logger1.error("This is an error message.")
    logger1.critical("This is a critical message.")

    print(f"\nCheck the log file: {DEFAULT_LOG_FILE}")

    # Get another logger with different settings
    logger2 = setup_logger(
        name='data_loader', log_level='DEBUG', log_file='data_loading.log')
    logger2.debug("Debug message from data_loader.")
    logger2.info("Info message from data_loader.")

    print(f"Check the log file: data_loading.log")

    # Get the first logger again (should return the same instance)
    logger1_again = setup_logger()
    print(
        f"\nIs logger1 the same as logger1_again? {logger1 is logger1_again}")
    logger1_again.info("Another info message from the default logger.")

    print("\nLogger test finished.")
