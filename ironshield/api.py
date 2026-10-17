"""IronShield REST API (FastAPI). Run on LinuxONE with:

    uvicorn ironshield.api:app --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .engine import Engine, split
from .model import RiskModel
from .stream import Streamer

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = RiskModel.load()
    engine = Engine(model)
    df = pd.read_csv(config.DATA_DIR / "transactions.csv")
    train, val, test = split(df)
    engine.warm_up(pd.concat([train, val]))
    streamer = Streamer(engine, test, rate=config.STREAM_RATE)
    STATE.update(engine=engine, streamer=streamer)
    if os.getenv("IRONSHIELD_AUTOSTREAM", "1") == "1":
        streamer.start()
    yield
    streamer.stop()


app = FastAPI(title="IronShield", version="0.1.0", lifespan=lifespan,
              description="Real-time, explainable fraud and mule-account detection on IBM Z / LinuxONE")


class Txn(BaseModel):
    ts: float
    type: str
    amount: float
    src: str
    dst: str
    src_old_bal: float = 0.0
    src_new_bal: float = 0.0
    dst_old_bal: float = 0.0
    dst_new_bal: float = 0.0
    new_device: int = 0


class Feedback(BaseModel):
    alert_id: int
    decision: str  # "block" or "approve"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(txn: Txn):
    """Score a single transaction (raw IDs are pseudonymised before analysis)."""
    data = txn.model_dump() if hasattr(txn, "model_dump") else txn.dict()  # pydantic v2 or v1
    return STATE["engine"].process(data)


@app.get("/stats")
def stats():
    s = STATE["engine"].stats()
    st = STATE["streamer"]
    s.update(stream_running=st.running, stream_rate=st.rate, sim_ts=st.sim_ts)
    return s


@app.get("/alerts")
def alerts(limit: int = 50, status: Optional[str] = None):
    return STATE["engine"].get_alerts(limit, status)


@app.get("/feed")
def feed(limit: int = 50):
    return STATE["engine"].get_feed(limit)


@app.get("/alerts/{alert_id}/network")
def network(alert_id: int):
    try:
        return STATE["engine"].ring_for(alert_id)
    except KeyError:
        raise HTTPException(404, "alert not found")


@app.post("/feedback")
def feedback(fb: Feedback):
    if fb.decision not in ("block", "approve"):
        raise HTTPException(400, "decision must be 'block' or 'approve'")
    try:
        return STATE["engine"].feedback(fb.alert_id, fb.decision)
    except KeyError:
        raise HTTPException(404, "alert not found")


@app.post("/stream/{action}")
def stream_control(action: str, rate: Optional[float] = None):
    st = STATE["streamer"]
    if action == "start":
        st.start()
    elif action == "stop":
        st.stop()
    elif action == "speed" and rate:
        st.rate = rate
    else:
        raise HTTPException(400, "action must be start, stop or speed (with ?rate=)")
    return {"running": st.running, "rate": st.rate}


@app.post("/demo/inject-scam")
def inject_scam(victims: int = 4, mules: int = 3):
    try:
        return STATE["streamer"].inject_scam(victims=victims, mules=mules)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
