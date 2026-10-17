"""Synthetic payment stream in the PaySim schema, with planted scam and mule-ring activity.

Normal life: customers pay merchants, transfer to a few regular contacts, cash in
and cash out, mostly during the day. A handful of legitimate "popular payees"
(landlords, small shops) receive from many people, and some customers make
legitimately large transfers, so the model cannot just learn "big = fraud".

Scam campaigns: victims are tricked into sending most of their balance to a
mule account in a ring. Mules forward almost everything to a collector within
minutes, and the collector cashes out. Victim transfers, mule forwards and the
collector cash-out are all labelled as fraud.
"""
import heapq
import math

import numpy as np
import pandas as pd

MIN = 60
HOUR = 3600
DAY = 24 * HOUR


def _day_time(rng, day):
    """A timestamp on `day`, weighted towards waking hours."""
    hour = rng.choice(24, p=_HOUR_WEIGHTS)
    return day * DAY + hour * HOUR + rng.uniform(0, HOUR)


_raw = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 1, 2, 4, 6, 6, 6, 7, 7, 6, 6, 6, 6, 7, 7, 6, 4, 2, 1])
_HOUR_WEIGHTS = _raw / _raw.sum()


def generate(n_customers=6000, n_merchants=400, days=5, n_rings=30, n_campaigns=90,
             n_popular=40, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = [f"C{100000 + i}" for i in range(n_customers)]
    merchants = [f"M{500000 + i}" for i in range(n_merchants)]
    balance = {c: float(rng.lognormal(8.0, 1.0)) for c in customers}
    typical = {c: float(rng.lognormal(4.0, 0.8)) for c in customers}
    rate = {c: float(rng.gamma(2.0, 1.5)) for c in customers}          # transactions per day
    popular = list(rng.choice(customers, n_popular, replace=False))
    contacts = {c: list(rng.choice(customers, 3, replace=False)) for c in customers}
    for c in customers:                                               # many people pay the popular payees
        if rng.random() < 0.15:
            contacts[c].append(popular[rng.integers(len(popular))])

    # Mule rings: fresh accounts with little history, plus one collector per ring
    rings = []
    next_id = 100000 + n_customers
    for _ in range(n_rings):
        size = int(rng.integers(3, 6))
        mules = [f"C{next_id + k}" for k in range(size)]
        collector = f"C{next_id + size}"
        next_id += size + 1
        for a in mules + [collector]:
            balance[a] = float(rng.uniform(5, 200))
            typical[a] = float(rng.lognormal(3.5, 0.5))
        rings.append((mules, collector))

    events = []  # (ts, seq, event)
    seq = 0

    def push(ts, ev):
        nonlocal seq
        heapq.heappush(events, (ts, seq, ev))
        seq += 1

    # ---- normal behaviour ----
    for c in customers:
        for day in range(days):
            for _ in range(rng.poisson(rate[c])):
                r = rng.random()
                ts = _day_time(rng, day)
                if r < 0.55:
                    ev = {"type": "PAYMENT", "src": c, "dst": merchants[rng.integers(n_merchants)],
                          "amount": float(rng.lognormal(math.log(typical[c]), 0.5))}
                elif r < 0.72:
                    big = rng.random() < 0.03          # legitimate large transfer (rent, savings move)
                    ev = {"type": "TRANSFER", "src": c, "dst": contacts[c][rng.integers(len(contacts[c]))],
                          "frac": float(rng.uniform(0.3, 0.8)) if big else None,
                          "amount": float(rng.lognormal(math.log(typical[c]) + 0.7, 0.6))}
                elif r < 0.85:
                    ev = {"type": "CASH_OUT", "src": c, "dst": merchants[rng.integers(n_merchants)],
                          "amount": float(rng.lognormal(math.log(typical[c]) + 0.5, 0.6))}
                elif r < 0.90:
                    ev = {"type": "DEBIT", "src": c, "dst": merchants[rng.integers(n_merchants)],
                          "amount": float(rng.lognormal(math.log(typical[c]), 0.5))}
                else:
                    ev = {"type": "CASH_IN", "src": c, "dst": merchants[rng.integers(n_merchants)],
                          "amount": float(rng.lognormal(math.log(typical[c]) + 1.5, 0.6))}
                ev.update(kind="normal", new_device=int(rng.random() < 0.01))
                push(ts, ev)

    # Popular payees forward some income too (legitimate fan-in, slow fan-out)
    for p in popular:
        for day in range(days):
            push(day * DAY + 20 * HOUR + rng.uniform(0, 2 * HOUR),
                 {"kind": "normal", "type": "TRANSFER", "src": p, "dst": contacts[p][0],
                  "frac": float(rng.uniform(0.2, 0.5)), "amount": 0.0, "new_device": 0})

    # ---- scam campaigns ----
    for _ in range(n_campaigns):
        mules, collector = rings[rng.integers(len(rings))]
        day = int(rng.integers(0, days))
        if rng.random() < 0.4:   # night-time campaign
            t0 = day * DAY + rng.uniform(0, 5 * HOUR)
        else:
            t0 = _day_time(rng, day)
        ato = rng.random() < 0.4  # account takeover: fraudster on a new device
        for v in rng.choice(customers, int(rng.integers(3, 8)), replace=False):
            push(t0 + rng.uniform(0, 45 * MIN),
                 {"kind": "victim", "type": "TRANSFER", "src": v, "dst": mules[rng.integers(len(mules))],
                  "frac": float(rng.uniform(0.6, 0.98)), "amount": 0.0,
                  "new_device": int(ato or rng.random() < 0.1), "collector": collector})

    # ---- replay events in time order, tracking balances ----
    rows = []
    pending_forward = set()
    while events:
        ts, _, ev = heapq.heappop(events)
        src, dst = ev["src"], ev["dst"]
        sb, db = balance.get(src, 0.0), balance.get(dst, 0.0)
        if ev.get("frac"):
            amount = round(ev["frac"] * sb, 2)
        else:
            amount = round(ev["amount"], 2)
        if ev["type"] == "CASH_IN":
            s_new, d_new = sb + amount, db
        else:
            if amount < 1 or amount > sb:
                if ev["kind"] != "normal" or sb < 2:
                    continue
                amount = round(sb * rng.uniform(0.2, 0.6), 2)
            s_new, d_new = sb - amount, db + amount
        balance[src], balance[dst] = s_new, d_new
        fraud = int(ev["kind"] != "normal")
        rows.append({"ts": ts, "type": ev["type"], "amount": amount, "src": src, "dst": dst,
                     "src_old_bal": round(sb, 2), "src_new_bal": round(s_new, 2),
                     "dst_old_bal": round(db, 2), "dst_new_bal": round(d_new, 2),
                     "new_device": ev.get("new_device", 0), "is_fraud": fraud, "scenario": ev["kind"]})

        # Mule behaviour: forward almost everything quickly, then the collector cashes out
        if ev["kind"] == "victim" and dst not in pending_forward:
            pending_forward.add(dst)
            push(ts + rng.uniform(2 * MIN, 20 * MIN),
                 {"kind": "mule_forward", "type": "TRANSFER" if rng.random() < 0.7 else "CASH_OUT",
                  "src": dst, "dst": ev["collector"], "frac": float(rng.uniform(0.9, 0.98)),
                  "amount": 0.0, "new_device": 0, "collector": ev["collector"]})
        elif ev["kind"] == "mule_forward":
            pending_forward.discard(src)
            if dst not in pending_forward:
                pending_forward.add(dst)
                push(ts + rng.uniform(10 * MIN, 40 * MIN),
                     {"kind": "collector_cashout", "type": "CASH_OUT", "src": dst,
                      "dst": merchants[rng.integers(n_merchants)], "frac": float(rng.uniform(0.9, 0.98)),
                      "amount": 0.0, "new_device": 0})
        elif ev["kind"] == "collector_cashout":
            pending_forward.discard(src)

    df = pd.DataFrame(rows)
    df["ts"] = df["ts"].round(1)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def load_paysim(path) -> pd.DataFrame:
    """Load the public PaySim CSV (Kaggle) into IronShield's schema."""
    raw = pd.read_csv(path)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ts": raw["step"] * HOUR + rng.uniform(0, HOUR, len(raw)),
        "type": raw["type"], "amount": raw["amount"],
        "src": raw["nameOrig"], "dst": raw["nameDest"],
        "src_old_bal": raw["oldbalanceOrg"], "src_new_bal": raw["newbalanceOrig"],
        "dst_old_bal": raw["oldbalanceDest"], "dst_new_bal": raw["newbalanceDest"],
        "new_device": 0, "is_fraud": raw["isFraud"], "scenario": "paysim",
    })
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    from . import config

    ap = argparse.ArgumentParser(description="Generate the IronShield synthetic transaction dataset")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    data = generate(n_customers=args.customers, days=args.days, seed=args.seed)
    out = config.DATA_DIR / "transactions.csv"
    data.to_csv(out, index=False)
    print(f"Wrote {len(data):,} transactions ({data.is_fraud.mean():.2%} fraud) to {out}")
    print(data.scenario.value_counts().to_string())
