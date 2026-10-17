"""Measure end-to-end scoring latency and throughput (the numbers for the pitch).

Runs held-out transactions through the full engine (privacy, features, model,
mule graph, explanations) one at a time, exactly as in production.

    python -m scripts.benchmark --n 20000
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ironshield import config  # noqa: E402
from ironshield.engine import Engine, split  # noqa: E402
from ironshield.model import RiskModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="transactions to score")
    args = ap.parse_args()

    df = pd.read_csv(config.DATA_DIR / "transactions.csv")
    train, val, test = split(df)
    engine = Engine(RiskModel.load())
    engine.warm_up(pd.concat([train, val]))
    rows = test.head(args.n).to_dict("records")

    for r in rows[:200]:            # warm caches / JIT paths
        engine.process(r)
    engine.reset_state()
    engine.warm_up(pd.concat([train, val]))
    engine.latencies_ms.clear()

    lat = []
    t0 = time.perf_counter()
    for r in rows:
        s = time.perf_counter()
        engine.process(r)
        lat.append((time.perf_counter() - s) * 1000)
    wall = time.perf_counter() - t0
    lat = np.array(lat)
    st = engine.stats()
    out = {
        "machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
        "model_backend": engine.model.backend, "transactions": len(rows),
        "latency_ms_avg": round(float(lat.mean()), 3), "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 3), "latency_ms_p99": round(float(np.percentile(lat, 99)), 3),
        "throughput_tps_single_thread": round(len(rows) / wall, 1),
        "alerts_high": st["flagged_high"], "alerts_medium": st["flagged_medium"],
        "live_precision": st["live_precision"], "live_recall": st["live_recall"],
    }
    name = f"benchmark_{platform.machine().lower()}.json"
    (config.RESULTS_DIR / name).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Saved to results/{name}")


if __name__ == "__main__":
    main()
