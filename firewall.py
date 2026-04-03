"""
firewall.py  –  Cross-platform firewall enforcement for FireRL.

On Windows  → uses 'netsh advfirewall' (requires Administrator).
On Linux    → uses iptables (requires root / CAP_NET_ADMIN).

Supports three block durations driven by the RL agent:
  BLOCK_TEMP  → 5-minute block (action 1)
  BLOCK_PERM  → permanent block until manually unblocked (action 2)
"""

import os
import subprocess
import time
import threading
import logging
from datetime import datetime, timezone

log = logging.getLogger("firerl.firewall")

IS_WINDOWS   = os.name == "nt"
RULE_PREFIX  = "FIRERL_BLOCK_"
CHAIN_NAME   = "FIRERL"

_blocked_ips: dict[str, dict] = {}
_lock        = threading.Lock()
_stop_event  = threading.Event()
_thread      = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── init ──────────────────────────────────────────────────────────────────────

def init_chain() -> None:
    if IS_WINDOWS:
        try:
            r = subprocess.run(
                ["netsh", "advfirewall", "show", "allprofiles"],
                capture_output=True, text=True)
            if r.returncode == 0:
                log.info("Windows Firewall ready.")
            else:
                log.warning("Windows Firewall check failed – run as Administrator.")
        except FileNotFoundError:
            log.error("netsh not found.")
    else:
        # Linux: create dedicated iptables chain
        try:
            subprocess.run(["iptables", "-N", CHAIN_NAME],
                           capture_output=True, check=False)
            subprocess.run(["iptables", "-C", "INPUT", "-j", CHAIN_NAME],
                           capture_output=True, check=False)
            subprocess.run(["iptables", "-A", "INPUT", "-j", CHAIN_NAME],
                           capture_output=True, check=False)
            log.info("iptables chain %s ready.", CHAIN_NAME)
        except Exception as e:
            log.warning("iptables init failed: %s", e)


# ── block / unblock ───────────────────────────────────────────────────────────

def block_ip(ip: str, duration_seconds: int | None = 300) -> bool:
    with _lock:
        if ip in _blocked_ips:
            return True
        ok = _os_block(ip)
        if ok:
            expires = time.time() + duration_seconds if duration_seconds else None
            _blocked_ips[ip] = {
                "blocked_at": _utcnow_iso(),
                "expires_at": datetime.fromtimestamp(expires, tz=timezone.utc).isoformat() if expires else None,
                "seconds_remaining": duration_seconds,
                "duration": duration_seconds,
            }
            log.info("Blocked %s for %s seconds.", ip, duration_seconds or "∞")
        return ok


def unblock_ip(ip: str) -> bool:
    with _lock:
        if ip not in _blocked_ips:
            return False
        ok = _os_unblock(ip)
        if ok:
            del _blocked_ips[ip]
            log.info("Unblocked %s.", ip)
        return ok


def is_blocked(ip: str) -> bool:
    with _lock:
        return ip in _blocked_ips


def get_blocked_ips() -> list[dict]:
    with _lock:
        now = time.time()
        result = []
        for ip, m in _blocked_ips.items():
            entry = {"ip": ip, **m}
            if m.get("duration") and m["expires_at"]:
                # Recompute remaining seconds
                try:
                    exp = datetime.fromisoformat(m["expires_at"].replace("Z", "+00:00"))
                    entry["seconds_remaining"] = max(0, round((exp.timestamp() - now), 1))
                except Exception:
                    pass
            result.append(entry)
        return result


# ── OS-level enforcement ──────────────────────────────────────────────────────

def _os_block(ip: str) -> bool:
    try:
        if IS_WINDOWS:
            rule = RULE_PREFIX + ip.replace(".", "_")
            subprocess.run([
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule}", "dir=in", "action=block",
                f"remoteip={ip}", "enable=yes", "profile=any",
            ], capture_output=True, text=True, check=True)
        else:
            subprocess.run([
                "iptables", "-A", CHAIN_NAME,
                "-s", ip, "-j", "DROP",
            ], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("block_ip %s failed: %s", ip, e.stderr)
        return False
    except Exception as e:
        log.warning("block_ip %s (no firewall access): %s", ip, e)
        return True   # In dev/testing without admin rights, simulate success


def _os_unblock(ip: str) -> bool:
    try:
        if IS_WINDOWS:
            rule = RULE_PREFIX + ip.replace(".", "_")
            subprocess.run([
                "netsh", "advfirewall", "firewall", "delete", "rule",
                f"name={rule}",
            ], capture_output=True, text=True, check=True)
        else:
            subprocess.run([
                "iptables", "-D", CHAIN_NAME, "-s", ip, "-j", "DROP",
            ], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("unblock_ip %s failed: %s", ip, e.stderr)
        return False
    except Exception as e:
        log.warning("unblock_ip %s: %s", ip, e)
        return True


def flush_chain() -> None:
    """Remove all FireRL block rules."""
    with _lock:
        for ip in list(_blocked_ips.keys()):
            _os_unblock(ip)
        _blocked_ips.clear()
    try:
        if not IS_WINDOWS:
            subprocess.run(["iptables", "-F", CHAIN_NAME], capture_output=True)
    except Exception:
        pass
    log.info("Firewall chain flushed.")


# ── auto-expiry scheduler ────────────────────────────────────────────────────

def _unblock_loop() -> None:
    while not _stop_event.is_set():
        _stop_event.wait(timeout=10)
        to_unblock = []
        with _lock:
            now = time.time()
            for ip, meta in list(_blocked_ips.items()):
                if meta.get("expires_at"):
                    try:
                        exp = datetime.fromisoformat(
                            meta["expires_at"].replace("Z", "+00:00"))
                        if now >= exp.timestamp():
                            to_unblock.append(ip)
                    except Exception:
                        pass
        for ip in to_unblock:
            unblock_ip(ip)


def start_unblock_scheduler() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_unblock_loop, daemon=True, name="unblock-sched")
    _thread.start()
    log.info("Auto-unblock scheduler started.")


def stop_unblock_scheduler() -> None:
    _stop_event.set()
