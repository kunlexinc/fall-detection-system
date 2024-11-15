"""Rule-based detector.

Two jobs: the baseline the ML model must beat, and the deterministic fallback
the service uses when the model is unavailable or too slow.
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES

PEAK_G = 2.5
POST_STD_G = 0.25
ORIENTATION_DEG = 45.0

_I = {n: i for i, n in enumerate(FEATURE_NAMES)}


def predict_one(feats: np.ndarray) -> int:
    return int(
        feats[_I["peak_g"]] >= PEAK_G
        and feats[_I["post_std_g"]] <= POST_STD_G
        and feats[_I["orientation_change_deg"]] >= ORIENTATION_DEG
    )


def predict(X: np.ndarray) -> np.ndarray:
    return np.array([predict_one(row) for row in X])
