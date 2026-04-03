"""
packet_sniffer.py  –  Real-time packet capture + RL decision loop.

Flow per packet
───────────────
  1. Extract IP/TCP/UDP metadata via Scapy
  2. Update per-IP traffic stats in the classifier registry
  3. Classify attack type + threat score (5-rule engine)
  4. Encode 5-dimensional RL state
  5. Agent makes ε-greedy decision (ALLOW / BLOCK_TEMP / BLOCK_PERM)
  6. Compute reward and apply Bellman Q-update
  7. Enforce decision via OS firewall
  8. Persist packet + decision records to MongoDB
  9. Store in rolling in-memory buffer for the /traffic API

Runs in a daemon thread; start() is called once from app.py startup.
"""

import os
import time
import threading
import logging
import warnings
from collections import deque
from datetime import datetime, timezone

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="scapy")

try:
    from scapy.all import (sniff, IP, TCP, UDP, ICMP,
                            conf as scapy_conf, get_if_list, get_if_addr)
    scapy_conf.ipv6_enabled = False
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

import classifier
import firewall
from rl_agent import agent
from db import db

log = logging.getLogger("firerl.sniffer")

BUFFER_SIZE = 500

_packet_buffer: deque[dict] = deque(maxlen=BUFFER_SIZE)
_lock          = threading.Lock()
_rl_active     = threading.Event()
_rl_active.set()
_prev_state: dict[str, tuple] = {}


# ── interface resolution ──────────────────────────────────────────────────────

def _best_interface(preferred: str | None = None) -> str | None:
    if preferred:
        return preferred
    if not SCAPY_OK:
        return None
    if os.name != "nt":
        return None
    try:
        for iface in get_if_list():
            addr = get_if_addr(iface)
            if addr and not addr.startswith("127.") and not addr.startswith("169.254."):
                log.info("Auto-selected interface: %s (%s)", iface, addr)
                return iface
    except Exception as e:
        log.warning("Interface detection error: %s", e)
    return None


# ── packet parsing ────────────────────────────────────────────────────────────

def _parse(pkt) -> dict | None:
    if not SCAPY_OK or not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    proto = "OTHER"; sp = dp = 0; flags = ""
    size = len(pkt)
    if pkt.haslayer(TCP):
        proto = "TCP"; sp = pkt[TCP].sport; dp = pkt[TCP].dport
        try:
            flags = str(pkt[TCP].flags)
        except Exception:
            flags = ""
    elif pkt.haslayer(UDP):
        proto = "UDP"; sp = pkt[UDP].sport; dp = pkt[UDP].dport
    elif pkt.haslayer(ICMP):
        proto = "ICMP"
    return {"src_ip": ip.src, "dst_ip": ip.dst, "src_port": sp,
            "dst_port": dp, "protocol": proto, "flags": flags,
            "size": size, "raw_ts": time.time()}


# ── packet handler ────────────────────────────────────────────────────────────

def _handle(pkt) -> None:
    meta = _parse(pkt)
    if meta is None:
        return

    src = meta["src_ip"]
    if firewall.is_blocked(src):
        return

    # 1. Update classifier stats
    classifier.update(src, meta["dst_port"], meta["protocol"],
                      meta["size"], meta["flags"])

    # 2. Classify
    attack_type, threat_score = classifier.classify(src)
    stats  = classifier.get_stats(src)
    window = max(time.time() - stats.get("first_seen", time.time()), 1.0)
    pps    = stats.get("pps", 1.0)
    bps    = stats.get("bps", 0.0)
    ports  = stats.get("distinct_ports", 1)

    # 3. Encode state
    state = agent.encode_state(threat_score, pps, ports, bps, meta["dst_port"])

    # 4. RL decision
    action_id   = 0
    action_name = "ALLOW"
    reward      = 0.0

    if _rl_active.is_set():
        action_id   = agent.decide(state)
        reward      = agent.compute_reward(action_id, attack_type, threat_score)
        action_name = {0: "ALLOW", 1: "BLOCK_TEMP", 2: "BLOCK_PERM"}.get(action_id, "ALLOW")

        prev = _prev_state.get(src, state)
        agent.update(prev, action_id, reward, state, attack_type, threat_score)
        _prev_state[src] = state

    # 5. Enforce
    if action_id == 1:
        firewall.block_ip(src, duration_seconds=300)
        db.upsert_blocked(src, firewall.get_blocked_ips()[0] if firewall.get_blocked_ips() else {})
    elif action_id == 2:
        firewall.block_ip(src, duration_seconds=None)  # permanent
        db.upsert_blocked(src, {})

    # 6. Build record
    ts = datetime.fromtimestamp(meta["raw_ts"], tz=timezone.utc).isoformat()
    record = {
        "timestamp":    ts,
        "src_ip":       src,
        "dst_ip":       meta["dst_ip"],
        "src_port":     meta["src_port"],
        "dst_port":     meta["dst_port"],
        "protocol":     meta["protocol"],
        "size":         meta["size"],
        "attack_type":  attack_type,
        "threat_score": round(threat_score, 4),
        "action":       action_name,
        "reward":       round(reward, 2),
        "pps":          round(pps, 2),
        "bps":          round(bps, 2),
        "state":        list(state),
    }

    with _lock:
        _packet_buffer.append(record)

    # 7. Persist
    db.insert_packet(record)
    db.insert_decision({
        "timestamp":   ts,
        "src_ip":      src,
        "attack_type": attack_type,
        "threat_score":round(threat_score, 4),
        "action":      action_name,
        "action_id":   action_id,
        "reward":      round(reward, 2),
        "state":       list(state),
    })


# ── demo mode (no Scapy / no root) ───────────────────────────────────────────

def _demo_loop() -> None:
    """
    Generate synthetic packets for development/demo without Scapy.
    Mimics the same pipeline as _handle() with realistic traffic patterns.
    """
    import random
    ATTACK_PROFILES = [
        # (attack_type, dst_ports, pps_range, bps_range)
        ("dos",         [80, 443],          (1500, 8000),   (2_000_000, 15_000_000)),
        ("port_scan",   list(range(20, 200))[:20], (3, 20), (500, 5_000)),
        ("brute_force", [22, 3389, 21],     (5, 40),        (1_000, 10_000)),
        ("suspicious",  [4444, 31337],      (1, 5),         (50, 400)),
        ("normal",      [80, 443, 8080],    (1, 15),        (5_000, 200_000)),
    ]
    src_pool = [f"192.168.{r}.{c}" for r in range(1, 6) for c in range(1, 20)]

    while True:
        ip  = random.choice(src_pool)
        idx = random.choices(range(5), weights=[5, 8, 8, 9, 70])[0]
        atype, ports, pps_r, bps_r = ATTACK_PROFILES[idx]

        dp   = random.choice(ports)
        pps  = random.uniform(*pps_r)
        bps  = random.uniform(*bps_r)
        size = int(bps / max(pps, 1))
        proto = random.choice(["TCP", "UDP", "ICMP"])
        flags = "S" if proto == "TCP" and random.random() < 0.4 else ""

        classifier.update(ip, dp, proto, size, flags)
        attack_type, threat_score = classifier.classify(ip)
        stats  = classifier.get_stats(ip)
        ports_ = stats.get("distinct_ports", 1)
        bps_   = stats.get("bps", bps)
        pps_   = stats.get("pps", pps)

        state = agent.encode_state(threat_score, pps_, ports_, bps_, dp)

        if _rl_active.is_set():
            action_id   = agent.decide(state)
            reward      = agent.compute_reward(action_id, attack_type, threat_score)
            action_name = {0: "ALLOW", 1: "BLOCK_TEMP", 2: "BLOCK_PERM"}.get(action_id, "ALLOW")
            prev = _prev_state.get(ip, state)
            agent.update(prev, action_id, reward, state, attack_type, threat_score)
            _prev_state[ip] = state
        else:
            action_id = 0; action_name = "ALLOW"; reward = 0.0

        if action_id == 1:
            firewall.block_ip(ip, duration_seconds=60)
        elif action_id == 2:
            firewall.block_ip(ip, duration_seconds=None)

        ts = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp":    ts,
            "src_ip":       ip,
            "dst_ip":       f"10.0.0.{random.randint(1,5)}",
            "src_port":     random.randint(49152, 65535),
            "dst_port":     dp,
            "protocol":     proto,
            "size":         size,
            "attack_type":  attack_type,
            "threat_score": round(threat_score, 4),
            "action":       action_name,
            "reward":       round(reward, 2),
            "pps":          round(pps_, 2),
            "bps":          round(bps_, 0),
            "state":        list(state),
        }
        with _lock:
            _packet_buffer.append(record)
        db.insert_packet(record)
        db.insert_decision({
            "timestamp":    ts,
            "src_ip":       ip,
            "attack_type":  attack_type,
            "threat_score": round(threat_score, 4),
            "action":       action_name,
            "action_id":    action_id,
            "reward":       round(reward, 2),
            "state":        list(state),
        })

        time.sleep(random.uniform(0.05, 0.25))


# ── public API ────────────────────────────────────────────────────────────────

def get_buffer(n: int = 100) -> list[dict]:
    with _lock:
        return list(_packet_buffer)[-n:]


def set_rl_mode(active: bool) -> None:
    if active:
        _rl_active.set()
    else:
        _rl_active.clear()


def rl_mode_is_active() -> bool:
    return _rl_active.is_set()


def start(interface: str | None = None) -> threading.Thread:
    demo = not SCAPY_OK or os.environ.get("FIRERL_DEMO", "0") == "1"
    if demo:
        log.info("Starting in DEMO mode (synthetic traffic generator).")
        t = threading.Thread(target=_demo_loop, daemon=True, name="demo-sniffer")
    else:
        iface = _best_interface(interface)
        log.info("Starting packet sniffer on %s.", iface or "default")
        def _loop():
            try:
                sniff(iface=iface, filter="ip", prn=_handle, store=False)
            except PermissionError:
                log.critical("Permission denied – run as Administrator / root.")
            except Exception as e:
                log.error("Sniffer error: %s – switching to demo mode.", e)
                _demo_loop()
        t = threading.Thread(target=_loop, daemon=True, name="pkt-sniffer")
    t.start()
    log.info("Sniffer thread started.")
    return t
