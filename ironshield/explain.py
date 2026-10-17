"""Turn model feature contributions into short, plain-English reasons."""
import math

from .features import FEATURES

_IDX = {name: i for i, name in enumerate(FEATURES)}


def _describe(name, x, txn):
    v = float(x[_IDX[name]])
    if name == "amount_vs_avg":
        return f"amount is {v:.0f}x this account's usual" if v >= 2 else None
    if name == "amount_log":
        return f"large amount ({txn['amount']:,.0f})"
    if name == "balance_drain":
        return f"moves {v:.0%} of the account balance" if v >= 0.3 else None
    if name == "empties_account":
        return "empties the account" if v else None
    if name == "src_tx_1h":
        return f"{v:.0f} payments from this account in the last hour" if v >= 2 else None
    if name == "mins_since_last":
        mins = math.expm1(v)
        if mins >= 7 * 24 * 60 - 1:
            return "account has no payment history"
        return f"only {mins:.0f} min since the previous payment" if mins < 30 else None
    if name == "src_history":
        return "brand-new or rarely used account" if v < 1.1 else None
    if name == "is_night":
        return f"unusual hour ({int(txn['ts'] // 3600) % 24:02d}:00)" if v else None
    if name == "new_device":
        return "made from a new device" if v else None
    if name == "new_payee":
        return "first ever payment to this receiver" if v else None
    if name == "dst_senders_1h":
        return f"receiver got money from {v:.0f} different senders in the last hour" if v >= 2 else None
    if name == "dst_history":
        return "receiver account has almost no history" if v < 1.1 else None
    if name == "forward_ratio":
        return f"forwards {min(v, 1):.0%} of money received in the last hour (mule pattern)" if v >= 0.5 else None
    if name == "src_senders_1h":
        return f"account was just paid by {v:.0f} different senders" if v >= 2 else None
    if name.startswith("type_"):
        return "cash-out: money leaves the banking system" if (v and name == "type_CASH_OUT") else None
    return None


def reasons(contrib, x, txn, top=3):
    """Top `top` human-readable reasons that pushed the risk up."""
    order = sorted(range(len(FEATURES)), key=lambda i: contrib[i], reverse=True)
    out = []
    for i in order:
        if contrib[i] <= 0 or len(out) >= top:
            break
        text = _describe(FEATURES[i], x, txn)
        if text and text not in out:
            out.append(text)
    return out
