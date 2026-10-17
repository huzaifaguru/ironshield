import numpy as np

from ironshield.features import FEATURES, FeatureState
from ironshield.mule import MuleDetector
from ironshield.privacy import protect, pseudonymise


def test_pseudonyms_are_stable_and_hide_ids():
    a = pseudonymise("C123")
    assert a == pseudonymise("C123")
    assert a != pseudonymise("C124")
    assert "C123" not in a
    safe = protect({"src": "C1", "dst": "C2", "amount": 5})
    assert safe["src"].startswith("acct_") and safe["dst"].startswith("acct_")


def _txn(ts, src, dst, amount, typ="TRANSFER", old=1000.0):
    return {"ts": ts, "type": typ, "src": src, "dst": dst, "amount": amount,
            "src_old_bal": old, "src_new_bal": old - amount, "new_device": 0}


def test_features_use_only_past_information():
    state = FeatureState()
    x1 = state.process(_txn(0, "A", "B", 100))
    x2 = state.process(_txn(60, "A", "B", 100))
    assert len(x1) == len(FEATURES)
    i = FEATURES.index("new_payee")
    assert x1[i] == 1 and x2[i] == 0          # B is only "new" the first time
    j = FEATURES.index("src_tx_1h")
    assert x1[j] == 0 and x2[j] == 1


def test_mule_detector_flags_fan_in_then_forward():
    det = MuleDetector()
    for k, victim in enumerate(["V1", "V2", "V3", "V4"]):
        det.add(_txn(k * 60, victim, "MULE", 900))
    fwd = _txn(600, "MULE", "BOSS", 3400, old=3600)
    det.add(fwd)
    res = det.check(fwd)
    assert res["mule_account"] == "MULE"
    assert res["mule_score"] >= 0.9
    ring = det.ring("MULE", 600, risky=0.0)
    assert {"V1", "BOSS", "MULE"} <= {n["id"] for n in ring["nodes"]}


def test_normal_shop_is_not_a_mule():
    det = MuleDetector()
    for k in range(10):                        # many customers pay a shop, shop keeps the money
        det.add(_txn(k * 60, f"C{k}", "SHOP", 20))
    s, _ = det.score("SHOP", 600)
    assert s < 0.5


def test_model_round_trip(tmp_path):
    from ironshield.model import RiskModel
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(FEATURES))).astype(np.float32)
    y = (X[:, 0] + X[:, 3] > 1.5).astype(int)
    m = RiskModel.train(X[:300], y[:300], X[300:], y[300:])
    m.save(tmp_path)
    m2 = RiskModel.load(tmp_path)
    assert np.allclose(m.predict(X[:5]), m2.predict(X[:5]))
    assert len(m2.contributions(X[0])) == len(FEATURES)
