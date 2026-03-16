# src/utils/logger.py
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "rag_agent") -> logging.Logger:
    """
    Sets up a centralized logger with Console and File output.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)

    # Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )

    # Handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Attach
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger