"""Minimal file-based model registry: versioned artifacts plus an 'active' pointer.

Rollback = point 'active' at an earlier version. No retraining, no redeploy.
(MLflow also records every run; this file is what the service reads.)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config as C


def _path(model_dir: Path) -> Path:
    return Path(model_dir) / C.REGISTRY_FILE


def load(model_dir: Path = C.MODEL_DIR) -> dict:
    p = _path(model_dir)
    return json.loads(p.read_text()) if p.exists() else {"active": None, "versions": []}


def save(reg: dict, model_dir: Path = C.MODEL_DIR) -> None:
    _path(model_dir).write_text(json.dumps(reg, indent=2))


def register(model_file: str, metrics: dict, params: dict,
             model_dir: Path = C.MODEL_DIR, activate: bool = True) -> int:
    reg = load(model_dir)
    version = max((v["version"] for v in reg["versions"]), default=0) + 1
    reg["versions"].append({
        "version": version,
        "file": model_file,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "params": params,
    })
    if activate:
        reg["active"] = version
    save(reg, model_dir)
    return version


def get(version: int, model_dir: Path = C.MODEL_DIR) -> dict:
    for v in load(model_dir)["versions"]:
        if v["version"] == version:
            return v
    raise KeyError(f"model version {version} not found")


def set_active(version: int, model_dir: Path = C.MODEL_DIR) -> None:
    get(version, model_dir)  # raises if missing
    reg = load(model_dir)
    reg["active"] = version
    save(reg, model_dir)
