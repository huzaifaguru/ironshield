"""IronShield analyst dashboard (Streamlit).

    streamlit run dashboard/app.py
Set IRONSHIELD_API to the LinuxONE server, e.g. http://148.100.x.x:8000
"""
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="IronShield | Live Fraud Command Center", page_icon="🛡️", layout="wide")

LEVEL_COLORS = {"HIGH": "#ffd7d9", "MEDIUM": "#fcf4d6", "LOW": "#ffffff"}

if "api" not in st.session_state:
    st.session_state.api = os.getenv("IRONSHIELD_API", "http://localhost:8000")


def call(method, path, **kw):
    try:
        r = requests.request(method, st.session_state.api.rstrip("/") + path, timeout=5, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.session_state.last_error = str(e)
        return None


def sim_clock(ts):
    day, rem = divmod(int(ts or 0), 86400)
    return f"Day {day + 1}, {rem // 3600:02d}:{(rem % 3600) // 60:02d}"


def platform_label(machine):
    return "IBM LinuxONE (s390x)" if machine == "s390x" else machine or "?"


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## 🛡️ IronShield")
    st.caption("Real-time, explainable fraud & mule detection on IBM Z / LinuxONE")
    st.session_state.api = st.text_input("API server", st.session_state.api)
    c1, c2 = st.columns(2)
    if c1.button("▶ Start", use_container_width=True):
        call("POST", "/stream/start")
    if c2.button("⏸ Pause", use_container_width=True):
        call("POST", "/stream/stop")
    rate = st.slider("Stream speed (tx/s)", 5, 200, 25, step=5)
    if st.button("Set speed", use_container_width=True):
        call("POST", "/stream/speed", params={"rate": rate})
    st.divider()
    st.markdown("### Demo")
    if st.button("🚨 Launch a live scam campaign", type="primary", use_container_width=True):
        res = call("POST", "/demo/inject-scam")
        if res:
            st.success(f"Scam launched: {res['victims']} victims → {res['mules']} mules → collector. Watch the alerts!")
    st.divider()
    st.caption("Account IDs shown are privacy-preserving pseudonyms (HMAC-SHA256). "
               "Raw identities never reach the AI.")


# ---------------- live view ----------------
@st.fragment(run_every="2s")
def live():
    stats = call("GET", "/stats")
    if stats is None:
        st.error(f"Cannot reach the IronShield API at {st.session_state.api}. "
                 f"{st.session_state.get('last_error', '')}")
        return

    st.markdown(f"### Live Fraud Command Center &nbsp; <span style='font-size:0.6em;color:#666'>"
                f"running on <b>{platform_label(stats['machine'])}</b> · simulated clock {sim_clock(stats['sim_ts'])}"
                f" · stream {'🟢 live' if stats['stream_running'] else '⏸ paused'}</span>", unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("Transactions scored", f"{stats['processed']:,}")
    m[1].metric("High-risk holds", f"{stats['flagged_high']:,}")
    m[2].metric("Avg decision time", f"{stats['latency_ms_avg']:.2f} ms")
    m[3].metric("p95 decision time", f"{stats['latency_ms_p95']:.2f} ms")
    m[4].metric("Capacity (1 core)", f"{stats['capacity_tps']:,} tx/s")
    m[5].metric("Precision / recall", f"{stats['live_precision'] * 100:.0f}/{stats['live_recall'] * 100:.0f}%")

    alerts = call("GET", "/alerts", params={"limit": 80}) or []
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("#### 🚨 Alerts awaiting review")
        open_alerts = [a for a in alerts if a["status"] == "open"]
        open_alerts.sort(key=lambda a: (a["level"] != "HIGH", -a["id"]))
        if not open_alerts:
            st.info("No open alerts right now.")
        for a in open_alerts[:4]:
            icon = "🔴" if a["level"] == "HIGH" else "🟠"
            with st.container(border=True):
                st.markdown(f"{icon} **{a['action']}** · {a['type'].replace('_', '-').lower()} of "
                            f"**{a['amount']:,.2f}** · risk **{a['risk']:.0%}** · decided in {a['latency_ms']:.1f} ms  \n"
                            f"<span style='font-size:0.8em'>`{a['src']}` → `{a['dst']}`</span>", unsafe_allow_html=True)
                st.caption(" · ".join(a["reasons"]))
                b1, b2, b3 = st.columns(3)
                if b1.button("⛔ Block", key=f"b{a['id']}", use_container_width=True):
                    call("POST", "/feedback", json={"alert_id": a["id"], "decision": "block"})
                    st.rerun(scope="fragment")
                if b2.button("✅ Approve", key=f"a{a['id']}", use_container_width=True):
                    call("POST", "/feedback", json={"alert_id": a["id"], "decision": "approve"})
                    st.rerun(scope="fragment")
                if b3.button("🕸 Network", key=f"n{a['id']}", use_container_width=True):
                    st.session_state.network_alert = a["id"]
        decided = [a for a in alerts if a["status"] != "open"]
        if decided:
            st.caption(f"Analyst decisions: {sum(a['status'] == 'blocked' for a in decided)} blocked, "
                       f"{sum(a['status'] == 'approved' for a in decided)} approved (saved as labels for retraining)")

    with right:
        st.markdown("#### 🕸 Mule network view")
        mule_alerts = [a for a in alerts if a["mule_score"] >= 0.5]
        target = st.session_state.get("network_alert") or (mule_alerts[0]["id"] if mule_alerts else None)
        render_network(target)

    st.markdown("#### Live transaction feed")
    feed = call("GET", "/feed", params={"limit": 12}) or []
    if feed:
        df = pd.DataFrame(feed)[["id", "type", "amount", "src", "dst", "risk", "level", "latency_ms"]]
        df["amount"] = df["amount"].map("{:,.2f}".format)
        styled = df.style.apply(lambda r: [f"background-color: {LEVEL_COLORS[r['level']]}; color: #0b1426"] * len(r),
                                axis=1)
        st.dataframe(styled, hide_index=True, use_container_width=True, height=420)


def render_network(target):
    if target is None:
        st.caption("No mule pattern detected yet. Launch a scam campaign from the sidebar to see one.")
        return
    net = call("GET", f"/alerts/{target}/network")
    if not net or not net["nodes"]:
        st.caption("The network for this alert has aged out of the live window.")
        return
    lines = ["digraph G {", 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];']
    for n in net["nodes"]:
        s = n["mule_score"]
        fill = "#da1e28" if s >= 0.5 else "#ff832b" if s >= 0.3 else "#e0e0e0"
        font = "white" if s >= 0.5 else "black"
        pen = ', penwidth=3, color="#0f62fe"' if n["focus"] else ""
        label = n["id"] + (f"\\nmule score {s:.2f}" if s >= 0.3 else "")
        lines.append(f'"{n["id"]}" [label="{label}", fillcolor="{fill}", fontcolor="{font}"{pen}];')
    for e in net["edges"]:
        color = "#da1e28" if e["risk"] >= 0.5 else "#8d8d8d"
        lines.append(f'"{e["src"]}" -> "{e["dst"]}" [label="{e["amount"]:,.0f}", color="{color}", '
                     f'fontsize=9, penwidth={2.5 if e["risk"] >= 0.5 else 1}];')
    lines.append("}")
    st.caption(f"Alert #{target}: the suspicious money trail. Red = mule pattern (many senders, money forwarded "
               "within minutes); labels are amounts; red arrows were scored high-risk.")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


live()
