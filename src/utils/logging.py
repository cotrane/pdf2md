"""Logging utility functions."""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: The logging level to use. Defaults to "INFO".
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pdf2md_{timestamp}.log"

    # Configure logging
    logging.basicConfig(  # pylint: disable=all
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
