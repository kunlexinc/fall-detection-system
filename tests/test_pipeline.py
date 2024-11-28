import importlib
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config as C
from src import synthetic, threshold
from src.data import load_dataset
from src.drift import psi
from src.features import FEATURE_NAMES, extract


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    synthetic.generate(d, n_fall=15, n_adl=10, seed=1)
    return d


@pytest.fixture(scope="session")
def model_dir(data_dir, tmp_path_factory):
    m = tmp_path_factory.mktemp("models")
    subprocess.run([sys.executable, "-m", "src.train", "--data", str(data_dir),
                    "--model-dir", str(m), "--no-mlflow"], check=True, cwd=C.ROOT)
    return m


@pytest.fixture()
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "events.db"))
    import app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_windows_have_expected_shape(data_dir):
    windows, hours = load_dataset(data_dir)
    assert windows and hours > 0
    assert all(w.samples.shape == (C.WINDOW_SAMPLES, 3) for w in windows)
    assert {w.label for w in windows} == {0, 1}


def test_features_and_rule_on_clear_fall(data_dir):
    windows, _ = load_dataset(data_dir)
    fall = next(w for w in windows if w.label == 1)
    f = extract(fall.samples)
    assert len(f) == len(FEATURE_NAMES)
    assert threshold.predict_one(f) == 1


def test_bad_window_rejected():
    with pytest.raises(ValueError):
        extract(np.zeros((10, 3)))


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    assert psi(ref, rng.normal(0, 1, 2000)) < 0.1
    assert psi(ref, rng.normal(2, 1, 2000)) > 0.25


def test_predict_feedback_metrics(client, data_dir):
    windows, _ = load_dataset(data_dir)
    fall = next(w for w in windows if w.label == 1)
    r = client.post("/predict", json={"samples": fall.samples.tolist(), "device_id": "t1"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_fall"] is True and body["source"] == "model"

    assert client.post("/feedback", json={"prediction_id": body["prediction_id"],
                                          "was_fall": True}).status_code == 200
    m = client.get("/metrics").json()
    assert m["requests"] == 1 and m["feedback_count"] == 1


def test_fallback_when_model_fails(client, data_dir, monkeypatch):
    import app.main as main
    windows, _ = load_dataset(data_dir)
    fall = next(w for w in windows if w.label == 1)

    def broken(_):
        raise RuntimeError("model crashed")
    monkeypatch.setattr(main, "_model_proba", broken)
    body = client.post("/predict", json={"samples": fall.samples.tolist()}).json()
    assert body["source"] == "fallback_rule"
    assert body["is_fall"] is True  # the alert still goes out


def test_rejects_wrong_length(client):
    assert client.post("/predict", json={"samples": [[0, 9.8, 0]] * 5}).status_code == 422


def test_rollback_unknown_version(client):
    assert client.post("/admin/rollback", json={"version": 999}).status_code == 404
