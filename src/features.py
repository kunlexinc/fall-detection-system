"""Hand-crafted features for a peak-aligned window.

A fall has a recognisable shape: a short free-fall dip (< 1 g), a hard impact,
then stillness in a new orientation. Each feature targets one of those phases.
"""
from __future__ import annotations

import numpy as np

from . import config as C

FEATURE_NAMES = [
    "peak_g",              # impact strength
    "pre_min_g",           # free-fall dip just before impact
    "mean_g",
    "std_g",
    "range_g",
    "max_jerk_g_per_s",    # how abrupt the impact is
    "post_std_g",          # stillness after impact
    "post_mean_g",
    "orientation_change_deg",  # lying vs standing
]


def _gravity_dir(xyz: np.ndarray) -> np.ndarray:
    v = xyz.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def extract(window: np.ndarray) -> np.ndarray:
    w = np.asarray(window, dtype=float)
    if w.shape != (C.WINDOW_SAMPLES, 3):
        raise ValueError(f"window must be ({C.WINDOW_SAMPLES}, 3), got {w.shape}")
    fs = C.SAMPLE_RATE_HZ
    mag = np.linalg.norm(w, axis=1) / C.G
    p = C.PEAK_INDEX
    pre = mag[max(0, p - int(0.5 * fs)):p + 1]
    post = mag[p + int(1.5 * fs):]                 # 1.5 s after impact onward

    before_dir = _gravity_dir(w[:int(0.8 * fs)])
    after_dir = _gravity_dir(w[-int(1.5 * fs):])
    cos = float(np.clip(np.dot(before_dir, after_dir), -1.0, 1.0))

    return np.array([
        mag.max(),
        pre.min(),
        mag.mean(),
        mag.std(),
        mag.max() - mag.min(),
        np.abs(np.diff(mag)).max() * fs,
        post.std(),
        post.mean(),
        np.degrees(np.arccos(cos)),
    ])


def extract_many(windows) -> np.ndarray:
    return np.vstack([extract(w) for w in windows])
