"""Lightweight logger wrapping Python logging and optional TensorBoard."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "CognitiveSemCom", level: int = logging.INFO,
                 log_file: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
