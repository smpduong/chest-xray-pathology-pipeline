"""Logging and I/O utilities."""

import logging
import sys
from pathlib import Path


def setup_logging(log_file: str = "output/pipeline.log") -> logging.Logger:
    """Configure dual file/console logging."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    return logging.getLogger("chest_xray_demo")
