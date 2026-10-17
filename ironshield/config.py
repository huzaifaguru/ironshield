"""Central settings for IronShield. Values can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("IRONSHIELD_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("IRONSHIELD_MODEL_DIR", ROOT / "models"))
RESULTS_DIR = Path(os.getenv("IRONSHIELD_RESULTS_DIR", ROOT / "results"))

# Secret used to pseudonymise account IDs. Set a real secret in production.
PRIVACY_SALT = os.getenv("IRONSHIELD_SALT", "ironshield-demo-salt-change-me")

# Rolling windows (seconds)
VELOCITY_WINDOW = 3600          # transactions per account in the last hour
MULE_WINDOW = 3600              # mule fan-in / fan-out window
GRAPH_RETENTION = 6 * 3600      # how long edges stay in the live transaction graph

# Mule detection thresholds
MULE_MIN_SENDERS = 3            # distinct senders in window before an account looks like a collector
MULE_FORWARD_RATIO = 0.7        # share of received money forwarded within the window

# Decision thresholds on the fused risk score
HIGH_RISK = float(os.getenv("IRONSHIELD_HIGH_RISK", "0.80"))
MEDIUM_RISK = float(os.getenv("IRONSHIELD_MEDIUM_RISK", "0.50"))

# Live stream replay speed (transactions per second)
STREAM_RATE = float(os.getenv("IRONSHIELD_STREAM_RATE", "25"))

for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
