"""Train and evaluate the fall detector, compare it with the rule baseline,
register a new model version, and log the run to MLflow.

    python -m src.train                        # real data in data/raw
    python -m src.train --data data/synthetic  # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from . import registry, threshold
from .data import load_dataset
from .features import FEATURE_NAMES, extract_many

MODELS = {
    # "large" is the accuracy-first starting point; "small" is the
    # latency-optimised version. Compare both with --model and src.benchmark.
    "rf_large": dict(n_estimators=500, max_depth=None),
    "rf_small": dict(n_estimators=50, max_depth=8),
}


def metrics(y_true, y_pred, adl_hours: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "recall": round(recall, 4),            # share of real falls caught
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "false_alarms": int(fp),
        "missed_falls": int(fn),
        "false_alarms_per_hour": round(fp / adl_hours, 3) if adl_hours else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def data_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.glob("*/*.csv")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def cross_validate(X, y, groups, params, n_splits=5, seed=42):
    """Out-of-fold predictions. Folds split by RECORDING, never by window, so
    windows from one recording can't leak between train and test."""
    n_splits = max(2, min(n_splits, len(set(groups[y == 1])), len(set(groups[y == 0]))))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model_pred = np.zeros_like(y)
    for tr, te in cv.split(X, y, groups):
        clf = RandomForestClassifier(**params, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        model_pred[te] = (clf.predict_proba(X[te])[:, 1] >= C.FALL_PROBA_THRESHOLD).astype(int)
    return model_pred, n_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=C.DATA_DIR)
    ap.add_argument("--model-dir", type=Path, default=C.MODEL_DIR)
    ap.add_argument("--model", choices=MODELS, default="rf_small")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    windows, adl_hours = load_dataset(args.data)
    X = extract_many([w.samples for w in windows])
    y = np.array([w.label for w in windows])
    groups = np.array([w.recording for w in windows])
    print(f"{len(y)} windows | falls={y.sum()} | ADL={len(y) - y.sum()} | "
          f"ADL hours={adl_hours:.2f} | recordings={len(set(groups))}")

    params = MODELS[args.model]
    model_pred, folds = cross_validate(X, y, groups, params)
    results = {
        "threshold_baseline": metrics(y, threshold.predict(X), adl_hours),
        args.model: metrics(y, model_pred, adl_hours),
    }
    print(f"\nOut-of-fold results ({folds}-fold, grouped by recording):")
    print(pd.DataFrame(results).T[["recall", "precision", "f1", "false_alarms",
                                   "missed_falls", "false_alarms_per_hour"]].to_string())

    # Final model on all data.
    clf = RandomForestClassifier(**params, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X, y)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_file = f"fall_model_{args.model}_{stamp}.joblib"
    joblib.dump({"model": clf, "features": FEATURE_NAMES, "name": args.model},
                args.model_dir / model_file)
    ref_file = model_file.replace(".joblib", "_reference.csv")
    pd.DataFrame(X, columns=FEATURE_NAMES).to_csv(args.model_dir / ref_file, index=False)

    fingerprint = data_fingerprint(args.data)
    version = registry.register(
        model_file,
        metrics={"cv": results[args.model], "baseline": results["threshold_baseline"]},
        params={**params, "model": args.model, "data": str(args.data),
                "data_fingerprint": fingerprint, "n_windows": int(len(y))},
        model_dir=args.model_dir,
    )
    print(f"\nRegistered model v{version}: {model_file} (now active)")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
            mlflow.set_experiment("fall-detection-system")
            with mlflow.start_run(run_name=f"{args.model}-v{version}"):
                mlflow.log_params({**params, "model": args.model, "registry_version": version,
                                   "data_fingerprint": fingerprint, "cv_folds": folds})
                for name, m in results.items():
                    for k in ("recall", "precision", "f1", "false_alarms"):
                        mlflow.log_metric(f"{name}.{k}", m[k])
                mlflow.log_artifact(str(args.model_dir / model_file))
            print("Logged run to MLflow (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db)")
        except ImportError:
            print("MLflow not installed; skipped experiment logging.")


if __name__ == "__main__":
    main()
