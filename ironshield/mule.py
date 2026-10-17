"""Mule-network detection on a live, rolling transaction graph.

A money mule typically (1) receives money from several unrelated senders in a
short time and (2) forwards most of it onward within minutes. The detector keeps
the last few hours of payments as a directed graph and scores both ends of every
new transaction for that fan-in / fast fan-out pattern.
"""
from collections import deque

import networkx as nx

from . import config


class MuleDetector:
    def __init__(self, window=config.MULE_WINDOW, retention=config.GRAPH_RETENTION):
        self.window = window
        self.retention = retention
        self.g = nx.DiGraph()
        self._edge_log = deque()          # (ts, u, v) for pruning
        self.history = {}                 # account -> total outgoing payments ever seen

    # ---------- graph maintenance ----------
    def _prune(self, now):
        while self._edge_log and now - self._edge_log[0][0] > self.retention:
            ts, u, v = self._edge_log.popleft()
            if self.g.has_edge(u, v):
                events = self.g[u][v]["events"]
                while events and events[0][0] <= ts:
                    events.popleft()
                if not events:
                    self.g.remove_edge(u, v)
                    for n in (u, v):
                        if self.g.degree(n) == 0:
                            self.g.remove_node(n)

    def add(self, txn, risk=0.0):
        u, v, ts = txn["src"], txn["dst"], txn["ts"]
        if txn["type"] == "CASH_IN":
            return
        self._prune(ts)
        if not self.g.has_edge(u, v):
            self.g.add_edge(u, v, events=deque())
        self.g[u][v]["events"].append((ts, float(txn["amount"]), float(risk)))
        self._edge_log.append((ts, u, v))
        self.history[u] = self.history.get(u, 0) + 1

    # ---------- scoring ----------
    def _window_flows(self, node, now):
        senders, inflow, outflow = set(), 0.0, 0.0
        if node not in self.g:
            return senders, inflow, outflow
        for p in self.g.predecessors(node):
            for ts, amt, _ in self.g[p][node]["events"]:
                if now - ts <= self.window:
                    senders.add(p)
                    inflow += amt
        for s in self.g.successors(node):
            for ts, amt, _ in self.g[node][s]["events"]:
                if now - ts <= self.window:
                    outflow += amt
        return senders, inflow, outflow

    def score(self, node, now):
        """Return (score 0..1, evidence text) for one account."""
        senders, inflow, outflow = self._window_flows(node, now)
        n = len(senders)
        fwd = min(outflow / inflow, 1.0) if inflow > 0 else 0.0
        fan_in = min(max(n - 1, 0) / (config.MULE_MIN_SENDERS - 1), 1.0)
        fast_out = min(fwd / config.MULE_FORWARD_RATIO, 1.0)
        new_account = self.history.get(node, 0) <= 3
        s = max(fan_in * (0.4 + 0.6 * fast_out), 0.45 * fast_out if (new_account and n >= 1) else 0.0)  # alone, stays below MEDIUM
        evidence = f"{n} sender(s) in the last {self.window // 60} min, {fwd:.0%} forwarded on"
        return s, evidence

    def check(self, txn):
        """Score both ends of a transaction (call after `add`). Returns the riskier one."""
        best = (0.0, None, "")
        for node in (txn["src"], txn["dst"]):
            s, ev = self.score(node, txn["ts"])
            if s > best[0]:
                best = (s, node, ev)
        return {"mule_score": round(best[0], 3), "mule_account": best[1], "mule_evidence": best[2]}

    # ---------- visualisation ----------
    def ring(self, account, now, max_nodes=40, risky=0.5):
        """The suspicious money trail around `account`, for the dashboard graph.

        Follows only high-risk payments (in either direction) so the picture shows
        victims -> mules -> collector -> cash-out, not every customer of a busy shop.
        """
        if account not in self.g:
            return {"nodes": [], "edges": []}
        recent = nx.DiGraph()
        for u, v, d in self.g.edges(data=True):
            ev = [e for e in d["events"] if now - e[0] <= self.window * 2]
            if ev:
                recent.add_edge(u, v, amount=sum(e[1] for e in ev), risk=max(e[2] for e in ev), count=len(ev))
        if account not in recent:
            return {"nodes": [], "edges": []}
        trail = nx.DiGraph((u, v, d) for u, v, d in recent.edges(data=True) if d["risk"] >= risky)
        if account in trail:
            keep = nx.node_connected_component(trail.to_undirected(as_view=True), account)
        else:
            keep = set(nx.ego_graph(recent.to_undirected(as_view=True), account, radius=1).nodes)
        keep = sorted(keep, key=lambda n: (n != account, -recent.degree(n)))[:max_nodes]
        sub = recent.subgraph(keep)
        nodes = []
        for n in sub.nodes:
            s, ev = self.score(n, now)
            nodes.append({"id": n, "mule_score": round(s, 3), "focus": n == account, "evidence": ev})
        edges = [{"src": u, "dst": v, "amount": round(d["amount"], 2), "risk": round(d["risk"], 3),
                  "count": d["count"]} for u, v, d in sub.edges(data=True)]
        return {"nodes": nodes, "edges": edges}
