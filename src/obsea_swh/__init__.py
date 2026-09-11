"""Inference utilities for the OBSEA significant-wave-height model."""

from .inference import predict_from_csv, predict_next

__all__ = ["predict_from_csv", "predict_next"]
__version__ = "1.0.0"
