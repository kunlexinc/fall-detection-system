"""Load phyphox accelerometer exports and cut them into peak-aligned windows.

Expected layout:
    data/raw/fall/*.csv   one fall per recording
    data/raw/adl/*.csv    activities of daily living (walking, sitting, stairs...)

phyphox experiment: "Acceleration (with g)" -> Export -> "CSV (Comma, decimal point)".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Window:
    samples: np.ndarray   # (WINDOW_SAMPLES, 3) in m/s^2
    label: int            # 1 = fall, 0 = ADL
    recording: str        # source file, used for group-aware splitting


def _find_col(df: pd.DataFrame, *needles: str) -> str:
    for col in df.columns:
        low = col.lower()
        if all(n in low for n in needles):
            return col
    raise ValueError(f"No column containing {needles} in {list(df.columns)}")


def load_recording(path: Path) -> tuple[np.ndarray, float]:
    """Return (xyz resampled to SAMPLE_RATE_HZ, duration_seconds)."""
    df = pd.read_csv(path)
    t = df[_find_col(df, "time")].to_numpy(dtype=float)
    axes = [df[_find_col(df, "acceleration", f" {a}")].to_numpy(dtype=float) for a in "xyz"]
    order = np.argsort(t)
    t = t[order]
    duration = float(t[-1] - t[0])
    t_new = np.arange(t[0], t[-1], 1.0 / C.SAMPLE_RATE_HZ)
    xyz = np.stack([np.interp(t_new, t, a[order]) for a in axes], axis=1)
    return xyz, duration


def magnitude_g(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz, axis=1) / C.G


def _cut(xyz: np.ndarray, peak: int) -> np.ndarray | None:
    start = peak - C.PEAK_INDEX
    end = start + C.WINDOW_SAMPLES
    if start < 0 or end > len(xyz):
        return None
    return xyz[start:end]


def candidate_peaks(xyz: np.ndarray, trigger_g: float = C.TRIGGER_G) -> list[int]:
    """Indices of local maxima above the trigger, at least MIN_EVENT_GAP_S apart."""
    mag = magnitude_g(xyz)
    gap = int(C.MIN_EVENT_GAP_S * C.SAMPLE_RATE_HZ)
    peaks: list[int] = []
    for i in np.argsort(mag)[::-1]:
        if mag[i] < trigger_g:
            break
        if all(abs(i - p) >= gap for p in peaks):
            peaks.append(int(i))
    return sorted(peaks)


def windows_from_recording(xyz: np.ndarray, label: int, name: str) -> list[Window]:
    if label == 1:
        # One fall per file: the impact is the largest peak.
        w = _cut(xyz, int(np.argmax(magnitude_g(xyz))))
        return [Window(w, 1, name)] if w is not None else []

    # ADL: every event that would have woken the trigger is a hard negative.
    out = [Window(w, 0, name) for p in candidate_peaks(xyz)
           if (w := _cut(xyz, p)) is not None]
    if not out:
        # Calm recording: still useful negatives, aligned on local maxima.
        step = C.WINDOW_SAMPLES
        for s in range(0, len(xyz) - step, step):
            seg = xyz[s:s + step]
            w = _cut(xyz, s + int(np.argmax(magnitude_g(seg))))
            if w is not None:
                out.append(Window(w, 0, name))
    return out


def load_dataset(root: Path = C.DATA_DIR) -> tuple[list[Window], float]:
    """Return (windows, total ADL hours recorded)."""
    windows: list[Window] = []
    adl_seconds = 0.0
    for label, sub in ((1, "fall"), (0, "adl")):
        for path in sorted((root / sub).glob("*.csv")):
            xyz, duration = load_recording(path)
            ws = windows_from_recording(xyz, label, f"{sub}/{path.name}")
            if label == 1 and not ws:
                print(f"[warn] {path.name}: fall peak too close to the edge, skipped")
            windows.extend(ws)
            if label == 0:
                adl_seconds += duration
    if not windows:
        raise SystemExit(f"No recordings found under {root}/fall and {root}/adl")
    return windows, adl_seconds / 3600
