"""Metaboleer: WGCNA modules + GGM + p-gain network pipeline for metabolomics."""
__version__ = "0.1.0"

from .config import Config  # noqa: E402,F401
from .pipeline import run_pipeline  # noqa: E402,F401
