"""Feature drift via Population Stability Index (PSI).

Compares live feature distributions with the training reference.
Rule of thumb: PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status(value: float) -> str:
    return "stable" if value < 0.1 else "watch" if value < 0.25 else "drift"


def report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    out = {}
    for col in reference.columns:
        if col in current.columns:
            v = psi(reference[col].to_numpy(), current[col].to_numpy())
            out[col] = {"psi": round(v, 4), "status": status(v)}
    return out
