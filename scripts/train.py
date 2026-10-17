"""Train the IronShield risk model and report test metrics.

    python -m scripts.train                 # synthetic data (generated if missing)
    python -m scripts.train --paysim PS.csv # public PaySim dataset instead
    python -m scripts.train --backend sklearn
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

from ironshield import config, datagen  # noqa: E402
from ironshield.engine import split  # noqa: E402
from ironshield.features import FEATURES, build_matrix  # noqa: E402
from ironshield.model import RiskModel  # noqa: E402


def metrics(y, p, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return {"precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4), "tp": tp, "fp": fp, "fn": fn,
            "test_transactions": int(len(y)), "test_fraud": int(y.sum())}


def load_feedback():
    path = config.DATA_DIR / "feedback.jsonl"
    if not path.exists():
        return None, None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return None, None
    return np.array([r["x"] for r in rows], dtype=np.float32), np.array([r["label"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paysim", help="path to the PaySim CSV from Kaggle")
    ap.add_argument("--backend", default="xgboost", choices=["xgboost", "sklearn"])
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset")
    args = ap.parse_args()

    data_path = config.DATA_DIR / "transactions.csv"
    if args.paysim:
        df = datagen.load_paysim(args.paysim)
        df.to_csv(data_path, index=False)
    elif args.regenerate or not data_path.exists():
        print("Generating synthetic dataset...")
        df = datagen.generate()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    print(f"{len(df):,} transactions, {df.is_fraud.mean():.2%} fraud")

    t0 = time.time()
    X, _ = build_matrix(df)
    y = df["is_fraud"].to_numpy()
    n_tr, n_val = len(split(df)[0]), len(split(df)[1])
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_val, y_val = X[n_tr:n_tr + n_val], y[n_tr:n_tr + n_val]
    X_te, y_te = X[n_tr + n_val:], y[n_tr + n_val:]
    print(f"Features built in {time.time() - t0:.1f}s")

    fb_X, fb_y = load_feedback()
    if fb_X is not None:
        print(f"Adding {len(fb_y)} analyst feedback labels")
        X_tr, y_tr = np.vstack([X_tr, fb_X]), np.concatenate([y_tr, fb_y])

    t0 = time.time()
    model = RiskModel.train(X_tr, y_tr, X_val, y_val, prefer=args.backend)
    train_secs = time.time() - t0
    model.save()

    m = metrics(y_te, model.predict(X_te), model.threshold)
    m.update(backend=model.backend, threshold=round(model.threshold, 3), train_seconds=round(train_secs, 2),
             machine=platform.machine(), platform=platform.platform(),
             dataset="paysim" if args.paysim else "synthetic")
    if model.backend == "xgboost":
        imp = model.model.feature_importances_
        m["top_features"] = [FEATURES[i] for i in np.argsort(imp)[::-1][:6]]
    (config.RESULTS_DIR / "model_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
