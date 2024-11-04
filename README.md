# Fall Detection: from phone sensor to production service

A prototype of a safety-critical fall-detection system: accelerometer data in,
a reliable alert decision out. It covers the full ML lifecycle: data
collection, feature engineering, a baseline vs. model comparison, low-latency
serving, a fail-safe fallback, versioning and rollback, production feedback,
and drift monitoring.

> **Status:** personal project, exploring what it takes to turn a simple
> sensor classifier into a reliable service. It is not deployed to real users.

## Why this problem

In a personal-safety product, the two errors cost different things:

| Error | Consequence |
|---|---|
| **Missed fall** (false negative) | Someone who needs help gets none |
| **False alarm** (false positive) | Watchers or responders get alerted for nothing. Too many and users switch the feature off |

So the system optimises **recall** while tracking **false alarms per hour**,
and it's designed to fail *safe*: a model problem must never stop an alert.

## Architecture

```
 Phone / wearable                         Inference service (FastAPI)
┌──────────────────────┐               ┌──────────────────────────────────────┐
│ accelerometer 50 Hz  │               │ features → rule detector (always on) │
│ cheap trigger:       │  4 s window   │          → ML model (100 ms budget)  │
│ |a| > 1.5 g          │ ────────────► │   timeout / error → rule decision    │
│ (saves battery: the  │               │ log: version, source, latency, feats │
│ model only sees      │ ◄──────────── │ alert if fall                        │
│ candidate events)    │   decision    └──────────────┬───────────────────────┘
└──────────────────────┘                              │
        ▲  "Are you OK?" answer                       ▼
        └──────── /feedback ──────────► SQLite → /metrics  /drift  /admin/rollback
```

**Event-triggered windowing.** A threshold on acceleration magnitude
(running on the device) picks out candidate events. Each one becomes a 4 s
window (1 s before the peak, 3 s after), so the model sees the whole fall
shape: free-fall dip → impact → stillness in a new orientation.

## What's implemented

| Area | Implementation |
|---|---|
| Data | Loader for phyphox CSV exports, resampled to 50 Hz, peak-aligned windows; ADL trigger events kept as *hard negatives* |
| Features | Peak impact, free-fall dip, jerk, post-impact stillness, orientation change (`src/features.py`) |
| Baseline | Interpretable threshold rule. The model has to beat it (`src/threshold.py`) |
| Model | Random forest; `rf_large` (accuracy-first) vs `rf_small` (latency-optimised) |
| Evaluation | Cross-validation **grouped by recording**, so windows from one session never appear in both train and test. Reports recall, precision, F1, missed falls, false alarms/hour |
| Serving | FastAPI `/predict` with a 100 ms model budget and **automatic fallback** to the rule |
| Versioning | File registry with an active-version pointer, data fingerprint, metrics per version; MLflow experiment tracking |
| Rollback | `POST /admin/rollback`, takes effect immediately with no redeploy |
| Production labels | `POST /feedback`: users answering "Are you OK?" give labels for false alarms and missed falls |
| Monitoring | `/metrics`: p50/p95/p99 latency, fallback rate, alert rate per model version, false-alarm share from feedback |
| Drift | `/drift`: Population Stability Index per feature, live traffic vs. training reference |
| Quality | Pytest suite (including the fallback path) and GitHub Actions CI |

## Results

**Latency** (`python -m src.benchmark`, single request, feature extraction +
inference, one CPU thread, laptop):

| Config | p50 | p95 | p99 | Fits 100 ms budget? |
|---|---|---|---|---|
| `rf_large` (500 trees) | 54.9 ms | 83.8 ms | 126.7 ms | No, p99 breaches it |
| `rf_small` (50 trees, depth 8) | 6.6 ms | 13.1 ms | 28.4 ms | Yes, ~4.5× headroom |

The large forest's tail latency would trigger the rule fallback on about 1
request in 100, so `rf_small` is the serving default.

**Accuracy.** The pipeline has so far been validated end to end on synthetic
recordings (`src.synthetic`), where both the rule and the model score
perfectly. That only proves the pipeline runs: the synthetic falls are too
clean to separate the two detectors. Results on real phone recordings
(see *Collecting data*) are pending.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test with synthetic data (pipeline check only; metrics are meaningless)
python -m src.synthetic --out data/synthetic
python -m src.train --data data/synthetic

# Real data
python -m src.train                    # trains on data/raw, registers a new version
python -m src.benchmark                # latency: large vs small model
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Serve
uvicorn app.main:app --reload          # docs at http://127.0.0.1:8000/docs
python scripts/simulate_traffic.py --n 200
python scripts/simulate_traffic.py --n 200 --shift 1.4   # simulate a miscalibrated phone → /drift flags it

pytest -q
```

## Collecting data

1. Install **phyphox** (free) and open the **"Acceleration (with g)"** experiment.
2. Keep the phone in the same place every time (e.g. front trouser pocket).
3. **Falls** → `data/raw/fall/`: one fall per recording. Stand still ~2 s, fall
   onto a mattress (forward, backward, sideways, from sitting), lie still 3–5 s, stop.
4. **Daily activity** → `data/raw/adl/`: walking, stairs, sitting down hard,
   lying down on purpose, jumping, putting the phone on a table.
5. Export **CSV (Comma, decimal point)** and save `Raw Data.csv` with a
   descriptive name (e.g. `fall_backward_03.csv`).

Only fall onto soft surfaces. Simulated falls differ from real ones (see Limitations).

## Design decisions

- **Rule first, model second.** The rule is the baseline the model has to
  beat, and the safety net when the model is unavailable.
- **Hard negatives.** Sitting down hard and jumping produce impact-like
  spikes. They are kept as negatives because they cause false alarms.
- **No leakage.** Overlapping windows from one recording are almost
  identical; splitting by window would inflate results. Splits are by recording.
- **Small model on purpose.** A shallow forest is fast, cheap, and simple to
  port to a microcontroller or phone. A neural network would have to justify its
  extra cost with measured gains.
- **Feedback as labels.** Labels are scarce in production. "I'm OK" responses
  are labelled false alarms, and manual panic presses with no automatic trigger
  before them point to missed falls.
- **Safe rollout (next step).** Run a new model in *shadow mode* alongside the
  active one, compare decisions on live traffic, then do a staged rollout with
  automatic rollback on a rise in false alarms or latency. Plain A/B testing
  is risky for an emergency feature.

## Limitations

- Simulated falls on a mattress are not real falls, especially for elderly
  users. Public datasets (e.g. SisFall) and real-world feedback would be needed.
- One phone, one wearing position, one person: device and placement drift are
  expected. That's why per-device monitoring exists.
- Runs server-side. In production, the trigger and probably the model should
  run on the device so detection works offline.

## Hardware path

Next step: deploy the trigger and model to an **Arduino Nano 33 BLE Sense**
(onboard IMU), with a Raspberry Pi gateway that relays alerts to this
service. Background: I spent a year as a lab attendant in a university
computer-engineering lab working on microcontroller and sensor projects.

## Credits

Inspired by Roni Bandini's Edge Impulse fall-detection project (Arduino Nano
33 BLE Sense + Raspberry Pi). This repo reimplements the ML pipeline in
Python and adds the production layer: serving, fallback, versioning,
monitoring and drift.
