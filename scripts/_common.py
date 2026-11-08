"""Shared helpers for the command-line scripts."""

from __future__ import annotations

import argparse
import logging
import os

THREADS = "4"


def limit_threads() -> None:
    """Pin BLAS/OpenMP threads; must run before numpy/torch are imported."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, THREADS)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
