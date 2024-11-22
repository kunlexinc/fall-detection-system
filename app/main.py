"""Fall-detection inference service.

    uvicorn app.main:app --reload

Design for a safety-critical path:
- The rule detector ALWAYS runs (cheap). If the model errors or exceeds
  MODEL_TIMEOUT_MS, the rule's decision is used. AI failure never blocks an alert.
- Every request is logged (version, source, latency, features) for monitoring.
- User feedback ("I'm OK" / "I really fell") becomes production labels.
- Rollback is a registry pointer change, applied without a redeploy.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config as C
from src import drift, registry, threshold
from src.features import FEATURE_NAMES, extract

MODEL_DIR = Path(os.getenv("MODEL_DIR", C.MODEL_DIR))
DB_PATH = Path(os.getenv("DB_PATH", C.ROOT / "events.db"))

state: dict = {"model": None, "version": None, "reference": None}


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY, ts REAL, device_id TEXT, model_version INTEGER,
        source TEXT, fall_proba REAL, is_fall INTEGER, latency_ms REAL,
        features TEXT, feedback INTEGER)""")
    return con


def load_active_model() -> None:
    reg = registry.load(MODEL_DIR)
    if reg["active"] is None:
        state.update(model=None, version=None, reference=None)
        return
    entry = registry.get(reg["active"], MODEL_DIR)
    bundle = joblib.load(MODEL_DIR / entry["file"])
    bundle["model"].set_params(n_jobs=1)
    ref = MODEL_DIR / entry["file"].replace(".joblib", "_reference.csv")
    state.update(model=bundle["model"], version=entry["version"],
                 reference=pd.read_csv(ref) if ref.exists() else None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db().close()
    load_active_model()
    yield


app = FastAPI(title="Fall Detection Service", lifespan=lifespan)


class PredictRequest(BaseModel):
    samples: list[list[float]] = Field(..., description="Nx3 accelerometer readings in m/s^2, "
                                       "peak-aligned, at 50 Hz (200 samples)")
    device_id: str = "unknown"


class Feedback(BaseModel):
    prediction_id: str
    was_fall: bool


class Rollback(BaseModel):
    version: int


def _model_proba(feats: np.ndarray) -> float:
    return float(state["model"].predict_proba(feats.reshape(1, -1))[0, 1])


@app.post("/predict")
async def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        feats = extract(np.array(req.samples))
    except ValueError as e:
        raise HTTPException(422, str(e))

    rule = threshold.predict_one(feats)
    source, proba = "fallback_rule", None
    if state["model"] is not None:
        try:
            proba = await asyncio.wait_for(asyncio.to_thread(_model_proba, feats),
                                           timeout=C.MODEL_TIMEOUT_MS / 1000)
            source = "model"
        except Exception:   # timeout or model error: degrade to the rule
            proba = None
    is_fall = int(proba >= C.FALL_PROBA_THRESHOLD) if proba is not None else rule
    latency = (time.perf_counter() - t0) * 1000

    pid = str(uuid.uuid4())
    with db() as con:
        con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,NULL)",
                    (pid, time.time(), req.device_id, state["version"], source, proba,
                     is_fall, latency, json.dumps(dict(zip(FEATURE_NAMES, feats.tolist())))))
    if is_fall:
        # In production this would notify watchers / the alarm centre via a queue.
        print(f"[ALERT] fall detected device={req.device_id} prediction={pid}")

    return {"prediction_id": pid, "is_fall": bool(is_fall), "fall_proba": proba,
            "source": source, "rule_says_fall": bool(rule),
            "model_version": state["version"], "latency_ms": round(latency, 2)}


@app.post("/feedback")
def feedback(fb: Feedback):
    with db() as con:
        cur = con.execute("UPDATE predictions SET feedback=? WHERE id=?",
                          (int(fb.was_fall), fb.prediction_id))
    if cur.rowcount == 0:
        raise HTTPException(404, "prediction not found")
    return {"ok": True}


@app.get("/metrics")
def metrics():
    with db() as con:
        df = pd.read_sql("SELECT * FROM predictions", con)
    if df.empty:
        return {"requests": 0}
    labelled = df.dropna(subset=["feedback"])
    out = {
        "requests": int(len(df)),
        "latency_ms": {f"p{q}": round(float(df.latency_ms.quantile(q / 100)), 2)
                       for q in (50, 95, 99)},
        "fallback_rate": round(float((df.source != "model").mean()), 4),
        "alert_rate": round(float(df.is_fall.mean()), 4),
        "by_model_version": df.groupby("model_version", dropna=False)
                              .agg(requests=("id", "count"), alert_rate=("is_fall", "mean"))
                              .reset_index().to_dict("records"),
        "feedback_count": int(len(labelled)),
    }
    if len(labelled):
        alerts = labelled[labelled.is_fall == 1]
        out["false_alarm_share_of_alerts"] = (
            round(float((alerts.feedback == 0).mean()), 4) if len(alerts) else None)
        out["missed_falls_reported"] = int(((labelled.is_fall == 0) & (labelled.feedback == 1)).sum())
    return out


@app.get("/drift")
def drift_check(min_requests: int = 50):
    if state["reference"] is None:
        raise HTTPException(404, "no reference data for the active model")
    with db() as con:
        rows = con.execute("SELECT features FROM predictions WHERE model_version=?",
                           (state["version"],)).fetchall()
    if len(rows) < min_requests:
        return {"status": "insufficient_data", "requests": len(rows), "needed": min_requests}
    current = pd.DataFrame([json.loads(r[0]) for r in rows])
    rep = drift.report(state["reference"], current)
    worst = max(rep.values(), key=lambda r: r["psi"])
    return {"model_version": state["version"], "requests": len(rows),
            "overall": worst["status"], "features": rep}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "model_version": state["version"]}


@app.get("/models")
def models():
    return registry.load(MODEL_DIR)


@app.post("/admin/rollback")
def rollback(body: Rollback):
    try:
        registry.set_active(body.version, MODEL_DIR)
    except KeyError as e:
        raise HTTPException(404, str(e))
    load_active_model()
    return {"active_version": state["version"]}
