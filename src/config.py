"""Central configuration. Training and serving import from here so they never disagree."""
from pathlib import Path

G = 9.81  # m/s^2

SAMPLE_RATE_HZ = 50          # everything is resampled to this rate
PRE_IMPACT_S = 1.0           # window seconds before the acceleration peak
POST_IMPACT_S = 3.0          # window seconds after the peak (captures stillness)
WINDOW_SAMPLES = int((PRE_IMPACT_S + POST_IMPACT_S) * SAMPLE_RATE_HZ)  # 200
PEAK_INDEX = int(PRE_IMPACT_S * SAMPLE_RATE_HZ)                         # 50

# Cheap always-on trigger (would run on the device). Only windows whose peak
# crosses this go to the model, which saves battery and compute.
TRIGGER_G = 1.5
MIN_EVENT_GAP_S = 2.0

# Decision threshold on the model's fall probability.
FALL_PROBA_THRESHOLD = 0.5

# Serving
MODEL_TIMEOUT_MS = 100       # past this, the service falls back to the rule

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REGISTRY_FILE = "registry.json"
