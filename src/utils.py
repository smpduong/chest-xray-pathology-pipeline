"""Logging and I/O utilities."""

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = "output/pipeline.log") -> logging.Logger:
    """Configure dual file/console logging."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("chest_xray_demo")
    logger.setLevel(logging.INFO)

    # Reconfigure cleanly if called more than once in a process.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
