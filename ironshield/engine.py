"""Decision engine: privacy -> features -> risk model + mule graph -> explanation -> alert."""
import itertools
import json
import platform
import threading
import time
from collections import deque

import numpy as np

from . import config
from .explain import reasons as explain_reasons
from .features import FeatureState
from .model import RiskModel
from .mule import MuleDetector
from .privacy import protect


def split(df, train=0.6, val=0.1):
    """Time-ordered split so the model is always tested on 'future' transactions."""
    n = len(df)
    a, b = int(n * train), int(n * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


class Engine:
    def __init__(self, model: RiskModel):
        self.model = model
        self.lock = threading.Lock()
        self._ids = itertools.count(1)
        self.reset_state()
        self.alerts = deque(maxlen=500)
        self.feed = deque(maxlen=200)
        self.latencies_ms = deque(maxlen=5000)
        self.processed = 0
        self.flagged = {"HIGH": 0, "MEDIUM": 0}
        self.confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.started = time.time()
        self.feedback_path = config.DATA_DIR / "feedback.jsonl"

    def reset_state(self):
        self.features = FeatureState()
        self.mules = MuleDetector()

    def warm_up(self, df):
        """Load past transactions into memory (no scoring) so accounts have history."""
        with self.lock:
            for r in df.to_dict("records"):
                safe = protect(r)
                self.features.update(safe)
                self.mules.add(safe)

    # ---------- scoring ----------
    def _scaled(self, p):
        """Map model probability so the model's decision threshold lands at HIGH_RISK."""
        t = self.model.threshold
        hi = config.HIGH_RISK
        return hi * p / t if p < t else hi + (1 - hi) * (p - t) / max(1 - t, 1e-9)

    def process(self, raw: dict) -> dict:
        with self.lock:
            t0 = time.perf_counter()
            safe = protect(raw)
            x = self.features.transform(safe)
            p = float(self.model.predict(x)[0])
            self.mules.add(safe, risk=p)
            mule = self.mules.check(safe)
            risk = max(self._scaled(p), mule["mule_score"])
            level = "HIGH" if risk >= config.HIGH_RISK else "MEDIUM" if risk >= config.MEDIUM_RISK else "LOW"
            why = []
            if level != "LOW":
                why = explain_reasons(self.model.contributions(x), x, safe)
                if mule["mule_score"] >= 0.5:
                    why.insert(0, "mule pattern: " + mule["mule_evidence"])
            self.features.update(safe)
            latency = (time.perf_counter() - t0) * 1000

            self.processed += 1
            self.latencies_ms.append(latency)
            result = {
                "id": next(self._ids), "ts": safe["ts"], "type": safe["type"], "amount": round(float(safe["amount"]), 2),
                "src": safe["src"], "dst": safe["dst"], "risk": round(risk, 3), "level": level,
                "model_prob": round(p, 4), **mule, "reasons": why[:4], "latency_ms": round(latency, 3),
            }
            truth = raw.get("is_fraud")
            if truth is not None:
                flagged = level == "HIGH"
                key = ("tp" if truth else "fp") if flagged else ("fn" if truth else "tn")
                self.confusion[key] += 1
            self.feed.append(result)
            if level != "LOW":
                self.flagged[level] += 1
                self.alerts.append({**result, "status": "open", "action": "HOLD" if level == "HIGH" else "REVIEW",
                                    "_x": x.tolist(), "_truth": truth})
            return result

    # ---------- analyst actions ----------
    def feedback(self, alert_id: int, decision: str) -> dict:
        with self.lock:
            for a in self.alerts:
                if a["id"] == alert_id:
                    a["status"] = "blocked" if decision == "block" else "approved"
                    with open(self.feedback_path, "a") as fh:
                        fh.write(json.dumps({"x": a["_x"], "label": int(decision == "block"),
                                             "alert_id": alert_id}) + "\n")
                    return self._public(a)
        raise KeyError(alert_id)

    # ---------- read APIs ----------
    @staticmethod
    def _public(a):
        return {k: v for k, v in a.items() if not k.startswith("_")}

    def get_alerts(self, limit=50, status=None):
        with self.lock:
            items = [self._public(a) for a in reversed(self.alerts) if status in (None, a["status"])]
        return items[:limit]

    def get_feed(self, limit=50):
        with self.lock:
            return list(reversed(self.feed))[:limit]

    def ring_for(self, alert_id: int):
        with self.lock:
            alert = next((a for a in self.alerts if a["id"] == alert_id), None)
            if alert is None:
                raise KeyError(alert_id)
            focus = alert["mule_account"] or alert["dst"]
            return {"focus": focus, **self.mules.ring(focus, alert["ts"])}

    def stats(self):
        with self.lock:
            lat = np.array(self.latencies_ms) if self.latencies_ms else np.zeros(1)
            c = self.confusion
            elapsed = max(time.time() - self.started, 1e-9)
            return {
                "processed": self.processed, "flagged_high": self.flagged["HIGH"],
                "flagged_medium": self.flagged["MEDIUM"],
                "latency_ms_avg": round(float(lat.mean()), 3), "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
                "capacity_tps": round(1000.0 / max(float(lat.mean()), 1e-6)),
                "observed_tps": round(self.processed / elapsed, 1),
                "live_precision": round(c["tp"] / max(c["tp"] + c["fp"], 1), 3),
                "live_recall": round(c["tp"] / max(c["tp"] + c["fn"], 1), 3),
                "model_backend": self.model.backend, "model_threshold": round(self.model.threshold, 3),
                "machine": platform.machine(), "platform": platform.platform(),
                "graph_nodes": self.mules.g.number_of_nodes(), "graph_edges": self.mules.g.number_of_edges(),
            }
