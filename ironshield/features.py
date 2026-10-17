"""Streaming feature engine.

Keeps a small rolling memory per (pseudonymised) account and turns each incoming
transaction into a feature vector using only information available at that
moment, so training and live scoring see exactly the same features.
"""
import math
from collections import defaultdict, deque

import numpy as np

from . import config

TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
FEATURES = [
    "amount_log", "amount_vs_avg", "src_history", "balance_drain", "empties_account",
    "src_tx_1h", "mins_since_last", "is_night", "new_device", "new_payee",
    "dst_senders_1h", "dst_history", "forward_ratio", "src_senders_1h",
] + [f"type_{t}" for t in TYPES]

NO_HISTORY_MINS = math.log1p(7 * 24 * 60)


class _Account:
    __slots__ = ("out_count", "out_sum", "last_out", "out_times", "in_count", "inflow", "payees")

    def __init__(self):
        self.out_count = 0
        self.out_sum = 0.0
        self.last_out = None
        self.out_times = deque()
        self.in_count = 0
        self.inflow = deque()        # (ts, sender, amount)
        self.payees = set()


def _prune(dq, now, window, key=lambda x: x):
    while dq and now - key(dq[0]) > window:
        dq.popleft()


class FeatureState:
    def __init__(self):
        self.accounts = defaultdict(_Account)

    def transform(self, txn: dict) -> np.ndarray:
        """Feature vector for `txn` given the history seen so far (does not update state)."""
        ts, amount = txn["ts"], float(txn["amount"])
        src, dst = self.accounts[txn["src"]], self.accounts[txn["dst"]]
        _prune(src.out_times, ts, config.VELOCITY_WINDOW)
        _prune(src.inflow, ts, config.VELOCITY_WINDOW, key=lambda x: x[0])
        _prune(dst.inflow, ts, config.VELOCITY_WINDOW, key=lambda x: x[0])

        outgoing = txn["type"] != "CASH_IN"
        avg = src.out_sum / src.out_count if src.out_count else None
        old_bal = float(txn.get("src_old_bal", 0) or 0)
        new_bal = float(txn.get("src_new_bal", 0) or 0)
        inflow_1h = sum(a for _, _, a in src.inflow)
        hour = int(ts // 3600) % 24

        f = [
            math.log1p(amount),
            min(amount / avg, 100.0) if avg else 1.0,
            math.log1p(src.out_count),
            min(amount / old_bal, 1.0) if (outgoing and old_bal > 0) else 0.0,
            float(outgoing and old_bal > 0 and new_bal < 0.05 * old_bal),
            float(len(src.out_times)),
            math.log1p((ts - src.last_out) / 60) if src.last_out is not None else NO_HISTORY_MINS,
            float(hour < 6),
            float(txn.get("new_device", 0)),
            float(txn["dst"] not in src.payees),
            float(len({s for _, s, _ in dst.inflow} | {txn["src"]})),
            math.log1p(dst.in_count),
            min(amount / inflow_1h, 2.0) if inflow_1h > 0 else 0.0,
            float(len({s for _, s, _ in src.inflow})),
        ] + [float(txn["type"] == t) for t in TYPES]
        return np.asarray(f, dtype=np.float32)

    def update(self, txn: dict) -> None:
        ts, amount = txn["ts"], float(txn["amount"])
        src, dst = self.accounts[txn["src"]], self.accounts[txn["dst"]]
        if txn["type"] != "CASH_IN":
            src.out_count += 1
            src.out_sum += amount
            src.last_out = ts
            src.out_times.append(ts)
            src.payees.add(txn["dst"])
            dst.in_count += 1
            dst.inflow.append((ts, txn["src"], amount))

    def process(self, txn: dict) -> np.ndarray:
        x = self.transform(txn)
        self.update(txn)
        return x


def build_matrix(df):
    """Run a whole (time-sorted) DataFrame through a fresh FeatureState."""
    state = FeatureState()
    rows = df[["ts", "type", "amount", "src", "dst", "src_old_bal", "src_new_bal", "new_device"]].to_dict("records")
    return np.vstack([state.process(r) for r in rows]), state
