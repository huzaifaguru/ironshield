"""Fraud risk model: XGBoost when available, scikit-learn otherwise.

Both backends expose the same interface: `predict(X)` returns fraud probabilities
and `contributions(x)` returns a per-feature contribution for one transaction,
which the explanation layer turns into plain-English reasons. XGBoost gives exact
TreeSHAP values natively, so no extra SHAP dependency is needed on LinuxONE.
"""
import json
import pickle

import numpy as np

from . import config
from .features import FEATURES

try:
    import xgboost as xgb
except ImportError:  # e.g. no s390x wheel on the server
    xgb = None


class RiskModel:
    def __init__(self, backend, model, threshold=0.5, baseline=None):
        self.backend = backend
        self.model = model
        self.threshold = threshold
        self.baseline = baseline  # per-feature medians, used for sklearn contributions

    # ---------- training ----------
    @classmethod
    def train(cls, X, y, X_val=None, y_val=None, prefer="xgboost"):
        pos = max(int(y.sum()), 1)
        weight = (len(y) - pos) / pos
        if prefer == "xgboost" and xgb is not None:
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                colsample_bytree=0.9, scale_pos_weight=weight, eval_metric="aucpr",
                tree_method="hist", n_jobs=4)
            model.fit(X, y)
            rm = cls("xgboost", model)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                                   class_weight={0: 1.0, 1: weight})
            model.fit(X, y)
            rm = cls("sklearn", model, baseline=np.median(X, axis=0).astype(np.float32))
        if X_val is not None:
            rm.threshold = rm._best_threshold(X_val, y_val)
        return rm

    def _best_threshold(self, X, y):
        p = self.predict(X)
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            pred = p >= t
            tp = float((pred & (y == 1)).sum())
            prec = tp / max(pred.sum(), 1)
            rec = tp / max((y == 1).sum(), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        return best_t

    # ---------- inference ----------
    def predict(self, X):
        X = np.atleast_2d(X)
        if self.backend == "xgboost":
            # inplace_predict skips the scikit-learn wrapper and DMatrix construction: much lower latency
            if not hasattr(self, "_booster"):
                self._booster = self.model.get_booster()
            return self._booster.inplace_predict(X, validate_features=False)
        return self.model.predict_proba(X)[:, 1]

    def contributions(self, x):
        """Per-feature contribution to this transaction's risk (positive = more risky)."""
        x = np.atleast_2d(x).astype(np.float32)
        if self.backend == "xgboost":
            contrib = self.model.get_booster().predict(xgb.DMatrix(x, feature_names=FEATURES),
                                                       pred_contribs=True)
            return contrib[0, :-1]  # last column is the bias term
        # sklearn: how much the probability drops when each feature is reset to a typical value
        base = self.predict(x)[0]
        probes = np.repeat(x, len(FEATURES), axis=0)
        probes[np.arange(len(FEATURES)), np.arange(len(FEATURES))] = self.baseline
        return base - self.predict(probes)

    # ---------- persistence ----------
    def save(self, directory=config.MODEL_DIR):
        meta = {"backend": self.backend, "threshold": self.threshold, "features": FEATURES}
        if self.backend == "xgboost":
            self.model.save_model(str(directory / "risk_model.json"))  # portable across CPU architectures
        else:
            with open(directory / "risk_model.pkl", "wb") as fh:
                pickle.dump({"model": self.model, "baseline": self.baseline}, fh)
        (directory / "risk_model_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory=config.MODEL_DIR):
        meta = json.loads((directory / "risk_model_meta.json").read_text())
        if meta["features"] != FEATURES:
            raise RuntimeError("Saved model was trained on a different feature set; retrain it.")
        if meta["backend"] == "xgboost":
            if xgb is None:
                raise RuntimeError("Model needs xgboost, which is not installed here; retrain with --backend sklearn.")
            model = xgb.XGBClassifier()
            model.load_model(str(directory / "risk_model.json"))
            return cls("xgboost", model, meta["threshold"])
        with open(directory / "risk_model.pkl", "rb") as fh:
            blob = pickle.load(fh)
        return cls("sklearn", blob["model"], meta["threshold"], blob["baseline"])
