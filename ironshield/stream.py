"""Simulated core-banking feed: replays transactions into the engine in real time.

It also tracks balances by raw account ID (as the bank's own system would) so a
live scam campaign can be injected on demand during the demo.
"""
import heapq
import itertools
import threading
import time

import numpy as np


class Streamer:
    def __init__(self, engine, df, rate=25.0, seed=11):
        self.engine = engine
        self.rows = df.to_dict("records")
        self.rate = rate
        self.rng = np.random.default_rng(seed)
        self.balance = {}
        self.injected = []                  # heap of (ts, seq, txn)
        self._seq = itertools.count()
        self._pos = 0
        self.sim_ts = self.rows[0]["ts"] if self.rows else 0.0
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        self.injected_campaigns = 0

    # ---------- control ----------
    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _next(self):
        with self._lock:
            data_ts = self.rows[self._pos]["ts"] if self._pos < len(self.rows) else float("inf")
            if self.injected and self.injected[0][0] <= data_ts:
                return heapq.heappop(self.injected)[2]
            if self._pos >= len(self.rows):
                return None
            row = self.rows[self._pos]
            self._pos += 1
            return row

    def _loop(self):
        while self.running:
            txn = self._next()
            if txn is None:                 # end of data: loop the replay with fresh state
                with self._lock:
                    self._pos = 0
                self.engine.reset_state()
                continue
            txn = self._apply_balance(txn)
            self.sim_ts = txn["ts"]
            self.engine.process(txn)
            time.sleep(1.0 / max(self.rate, 0.1))

    def _apply_balance(self, txn):
        txn = dict(txn)
        if txn.get("frac") is not None:     # injected scam steps take a share of the live balance
            src_bal = self.balance.get(txn["src"], 0.0)
            txn["amount"] = round(txn.pop("frac") * src_bal, 2)
            txn["src_old_bal"] = round(src_bal, 2)
            txn["src_new_bal"] = round(src_bal - txn["amount"], 2)
            txn["dst_old_bal"] = round(self.balance.get(txn["dst"], 0.0), 2)
            txn["dst_new_bal"] = round(txn["dst_old_bal"] + txn["amount"], 2)
        self.balance[txn["src"]] = txn["src_new_bal"]
        self.balance[txn["dst"]] = txn["dst_new_bal"]
        return txn

    # ---------- demo: inject a live scam campaign ----------
    def inject_scam(self, victims=4, mules=3, new_device=True):
        rich = [a for a, b in self.balance.items() if b > 800 and a.startswith("C")]
        if len(rich) < victims:
            raise RuntimeError("Not enough account history yet; let the stream run a few seconds.")
        t0 = self.sim_ts + 30
        self.injected_campaigns += 1
        tag = 9_000_000 + self.injected_campaigns * 10
        mule_ids = [f"C{tag + k}" for k in range(mules)]
        collector = f"C{tag + mules}"
        cash_agent = "M500001"
        plan = []
        for i, v in enumerate(self.rng.choice(rich, victims, replace=False)):
            plan.append((t0 + i * 90 + self.rng.uniform(0, 60), str(v), mule_ids[i % mules],
                         float(self.rng.uniform(0.75, 0.95)), "TRANSFER", int(new_device)))
        last_victim = max(p[0] for p in plan)
        for k, m in enumerate(mule_ids):
            plan.append((last_victim + 120 + k * 40, m, collector, 0.95, "TRANSFER", 0))
        plan.append((last_victim + 600, collector, cash_agent, 0.97, "CASH_OUT", 0))
        with self._lock:
            for ts, src, dst, frac, typ, dev in plan:
                heapq.heappush(self.injected, (ts, next(self._seq), {
                    "ts": ts, "type": typ, "src": src, "dst": dst, "frac": frac, "new_device": dev,
                    "is_fraud": 1, "scenario": "injected"}))
        return {"victims": victims, "mules": mules, "transactions": len(plan), "starts_at_sim_ts": t0}
