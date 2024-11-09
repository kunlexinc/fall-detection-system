"""Generate SYNTHETIC phyphox-style recordings.

For smoke-testing the pipeline and CI only. Metrics on this data mean nothing
about real-world performance; train on real phone recordings for that.

    python -m src.synthetic --out data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import G

FS = 100  # phones usually record faster than we model; the loader resamples


def _rot(v: np.ndarray, rng) -> np.ndarray:
    v = v + rng.normal(0, 0.15, 3)
    return v / np.linalg.norm(v)


def _walk(n, rng, up):
    t = np.arange(n) / FS
    bounce = 0.25 * np.sin(2 * np.pi * rng.uniform(1.6, 2.2) * t)
    return np.outer(1 + bounce, up) * G + rng.normal(0, 0.4, (n, 3))


def _still(n, rng, up):
    return np.outer(np.ones(n), up) * G + rng.normal(0, 0.08, (n, 3))


def fall_recording(rng) -> np.ndarray:
    up = _rot(np.array([0, 1.0, 0]), rng)
    lying = _rot(np.array([0, 0, 1.0]), rng)
    parts = [_walk(int(FS * rng.uniform(2, 4)), rng, up)]
    ff = int(FS * rng.uniform(0.25, 0.45))                      # free fall
    parts.append(np.outer(np.linspace(0.9, 0.2, ff), up) * G + rng.normal(0, 0.3, (ff, 3)))
    imp = int(FS * 0.08)                                        # impact
    parts.append(np.outer(np.full(imp, rng.uniform(3.0, 6.0)), _rot(up + lying, rng)) * G)
    parts.append(_still(int(FS * rng.uniform(4, 6)), rng, lying))  # lying still
    return np.vstack(parts)


def adl_recording(rng) -> np.ndarray:
    up = _rot(np.array([0, 1.0, 0]), rng)
    parts = []
    for _ in range(rng.integers(4, 8)):
        kind = rng.choice(["walk", "still", "sit", "jump"])
        n = int(FS * rng.uniform(3, 8))
        if kind == "walk":
            parts.append(_walk(n, rng, up))
        elif kind == "still":
            parts.append(_still(n, rng, up))
        else:  # hard negatives: sitting down hard, jumping (spike, but upright after)
            seg = _walk(n, rng, up)
            k = n // 2
            seg[k:k + 8] = np.outer(np.full(8, rng.uniform(1.8, 3.2)), up) * G
            parts.append(seg)
    return np.vstack(parts)


def _to_csv(xyz: np.ndarray, path: Path) -> None:
    t = np.arange(len(xyz)) / FS
    pd.DataFrame({
        "Time (s)": t,
        "Acceleration x (m/s^2)": xyz[:, 0],
        "Acceleration y (m/s^2)": xyz[:, 1],
        "Acceleration z (m/s^2)": xyz[:, 2],
        "Absolute acceleration (m/s^2)": np.linalg.norm(xyz, axis=1),
    }).to_csv(path, index=False)


def generate(out: Path, n_fall: int = 40, n_adl: int = 25, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    for sub, n, fn in (("fall", n_fall, fall_recording), ("adl", n_adl, adl_recording)):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _to_csv(fn(rng), d / f"synthetic_{sub}_{i:03d}.csv")
    print(f"Wrote {n_fall} fall + {n_adl} ADL synthetic recordings to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    generate(a.out, seed=a.seed)
