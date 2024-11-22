"""Send recorded windows to the running service to exercise /metrics and /drift.

    uvicorn app.main:app &
    python scripts/simulate_traffic.py --data data/raw --n 200
    python scripts/simulate_traffic.py --data data/raw --n 200 --shift 1.3   # fake sensor drift
"""
import argparse
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw"))
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shift", type=float, default=1.0,
                    help="multiply readings to mimic a miscalibrated new phone model")
    a = ap.parse_args()

    windows, _ = load_dataset(a.data)
    with httpx.Client(base_url=a.url, timeout=5) as c:
        for i in range(a.n):
            w = random.choice(windows)
            r = c.post("/predict", json={"samples": (w.samples * a.shift).tolist(),
                                         "device_id": f"sim-{i % 10}"}).json()
            if random.random() < 0.3:  # some users answer the "Are you OK?" prompt
                c.post("/feedback", json={"prediction_id": r["prediction_id"],
                                          "was_fall": bool(w.label)})
        print(c.get("/metrics").json())
        print(c.get("/drift").json().get("overall"))


if __name__ == "__main__":
    main()
