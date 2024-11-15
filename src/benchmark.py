"""Measure inference latency (feature extraction + model) for the large and
small model configs, so before/after numbers come from your own machine.

    python -m src.benchmark --data data/raw
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from .data import load_dataset
from .features import extract, extract_many
from .train import MODELS


def time_single_requests(clf, windows, n=300) -> dict:
    clf.set_params(n_jobs=1)  # one request at a time, like a live service
    lat = []
    for i in range(n):
        w = windows[i % len(windows)]
        t0 = time.perf_counter()
        clf.predict_proba(extract(w).reshape(1, -1))
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat[10:])  # drop warm-up
    return {"p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "p99_ms": round(float(np.percentile(lat, 99)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=C.DATA_DIR)
    args = ap.parse_args()
    windows, _ = load_dataset(args.data)
    samples = [w.samples for w in windows]
    X = extract_many(samples)
    y = np.array([w.label for w in windows])

    print(f"{'config':<10} {'p50_ms':>8} {'p95_ms':>8} {'p99_ms':>8}")
    for name, params in MODELS.items():
        clf = RandomForestClassifier(**params, class_weight="balanced", random_state=42)
        clf.fit(X, y)
        r = time_single_requests(clf, samples)
        print(f"{name:<10} {r['p50_ms']:>8} {r['p95_ms']:>8} {r['p99_ms']:>8}")
    print("\nCompare accuracy too: python -m src.train --model rf_large  vs  --model rf_small")


if __name__ == "__main__":
    main()
