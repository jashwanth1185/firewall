"""
app.py  –  FireRL Flask API server.

Endpoints
─────────
  GET  /api/traffic          – recent packets (live feed + DB history)
  GET  /api/logs             – firewall decision log from DB
  GET  /api/blocked          – currently blocked IPs
  GET  /api/model            – RL agent stats, Q-table, rule history
  GET  /api/stats            – aggregate packet/decision stats from DB
  GET  /api/rule-history     – dynamic threshold adaptation history
  POST /api/decision         – manual ALLOW/BLOCK override
  POST /api/rl-mode          – toggle RL auto-mode
  POST /api/unblock          – manually unblock an IP
  POST /api/reset-agent      – reset Q-table to zeros (fresh start)
  GET  /api/health           – health check (DB connectivity, RL status)
"""

import os
import signal
import sys
import logging
import threading
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

import firewall
import packet_sniffer
from rl_agent import agent
from db import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("firerl.app")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ── startup ───────────────────────────────────────────────────────────────────

def startup() -> None:
    db.connect()
    firewall.init_chain()
    firewall.start_unblock_scheduler()
    iface = os.environ.get("FIRERL_IFACE", None)
    packet_sniffer.start(interface=iface)

    # Periodic agent-stats snapshot to DB (every 60 s)
    def _snapshot_loop():
        while True:
            time.sleep(60)
            try:
                db.insert_agent_snapshot(agent.get_stats())
            except Exception:
                pass
    threading.Thread(target=_snapshot_loop, daemon=True, name="stats-snap").start()

    log.info("FireRL backend ready. MongoDB: %s", "connected" if db.ok else "unavailable")


def shutdown(sig, frame) -> None:
    log.info("Shutting down – saving Q-table and flushing firewall…")
    firewall.flush_chain()
    firewall.stop_unblock_scheduler()
    agent.save()
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ok(data: dict, **kw) -> tuple:
    return jsonify({"status": "ok", **data, **kw}), 200


def _err(msg: str, code: int = 400) -> tuple:
    return jsonify({"status": "error", "error": msg}), code


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return _ok({
        "rl_mode":    packet_sniffer.rl_mode_is_active(),
        "db":         db.ok,
        "agent":      {"epsilon": round(agent.epsilon, 4),
                       "total_decisions": agent.total_decisions},
        "blocked":    len(firewall.get_blocked_ips()),
    })


@app.get("/api/traffic")
def get_traffic():
    n    = min(int(request.args.get("n", 100)), 500)
    live = packet_sniffer.get_buffer(n)
    hist = db.get_packets(n) if db.ok else []
    return _ok({
        "rl_mode": packet_sniffer.rl_mode_is_active(),
        "packets": live,
        "db_packets": hist,
        "count":   len(live),
    })


@app.get("/api/logs")
def get_logs():
    n    = min(int(request.args.get("n", 100)), 500)
    logs = db.get_decisions(n)
    return _ok({"logs": logs, "count": len(logs)})


@app.get("/api/blocked")
def get_blocked():
    mem = firewall.get_blocked_ips()
    dbs = db.get_blocked() if db.ok else []
    return _ok({
        "blocked":        mem,
        "db_blocked":     dbs,
        "count":          len(mem),
    })


@app.get("/api/model")
def get_model():
    stats = agent.get_stats()
    return _ok({
        "rl_mode":    packet_sniffer.rl_mode_is_active(),
        "agent":      stats,
        "db_history": db.get_agent_history(50) if db.ok else [],
    })


@app.get("/api/stats")
def get_stats():
    pkt_stats  = db.get_packet_stats()   if db.ok else {}
    dec_summary = db.decision_summary()  if db.ok else {}
    return _ok({
        "packet_stats":   pkt_stats,
        "decision_summary": dec_summary,
    })


@app.get("/api/rule-history")
def get_rule_history():
    live = agent.rule_history[-50:]
    dbs  = db.get_rule_changes(50) if db.ok else []
    return _ok({"rule_history": live, "db_history": dbs})


@app.post("/api/decision")
def manual_decision():
    data     = request.get_json(force=True) or {}
    ip       = data.get("ip", "").strip()
    action   = data.get("action", "").upper()
    duration = int(data.get("duration", 300))

    if not ip or action not in ("ALLOW", "BLOCK_TEMP", "BLOCK_PERM"):
        return _err("Provide 'ip' and 'action' (ALLOW|BLOCK_TEMP|BLOCK_PERM)")

    if action == "ALLOW":
        ok = firewall.unblock_ip(ip)
        db.remove_blocked(ip)
    elif action == "BLOCK_TEMP":
        ok = firewall.block_ip(ip, duration_seconds=duration)
        if ok:
            db.upsert_blocked(ip, {"reason": "manual", "duration": duration})
    else:
        ok = firewall.block_ip(ip, duration_seconds=None)
        if ok:
            db.upsert_blocked(ip, {"reason": "manual_perm"})

    return _ok({"ip": ip, "action": action, "success": ok})


@app.post("/api/rl-mode")
def set_rl_mode():
    data   = request.get_json(force=True) or {}
    active = bool(data.get("active", True))
    packet_sniffer.set_rl_mode(active)
    return _ok({"rl_mode": packet_sniffer.rl_mode_is_active()})


@app.post("/api/unblock")
def manual_unblock():
    data = request.get_json(force=True) or {}
    ip   = data.get("ip", "").strip()
    if not ip:
        return _err("Provide 'ip'")
    ok = firewall.unblock_ip(ip)
    if ok:
        db.remove_blocked(ip)
    return _ok({"ip": ip, "unblocked": ok})


@app.post("/api/reset-agent")
def reset_agent():
    """Reset Q-table to zeros for a fresh training start."""
    import numpy as np
    with agent._lock:
        agent.q_table[:] = 0.0
        agent.epsilon     = 0.40
        agent.total_decisions = 0
        agent.correct         = 0
        agent.fp_count = agent.fn_count = agent.tp_count = agent.tn_count = 0
        agent.rewards_history.clear()
        agent.rule_history.clear()
    agent.save()
    log.info("Agent Q-table reset.")
    return _ok({"reset": True})


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
