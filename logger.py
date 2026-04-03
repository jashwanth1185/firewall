"""
logger.py - Centralized logging for FireRL.
Writes all ALLOW/BLOCK decisions with metadata to a rotating log file.
"""

import logging
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "firewall.log")

os.makedirs(LOG_DIR, exist_ok=True)

# ── file handler (JSON-lines, 5 MB × 3 backups) ──────────────────────────────
_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
_file_handler.setLevel(logging.INFO)

_logger = logging.getLogger("firerl")
_logger.setLevel(logging.INFO)
_logger.addHandler(_file_handler)

# ── in-memory ring buffer (last 500 entries) for the API ─────────────────────
_log_buffer: list[dict] = []
_MAX_BUFFER = 500


def record(
    ip: str,
    port: int,
    protocol: str,
    action: str,
    threat_score: float,
    attack_type: str = "unknown",
    reason: str = "",
) -> dict:
    """
    Write one decision record to the log file and the in-memory buffer.
    Returns the record dict so callers can use it directly.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ip": ip,
        "port": port,
        "protocol": protocol,
        "action": action,            # "ALLOW" | "BLOCK"
        "threat_score": round(threat_score, 4),
        "attack_type": attack_type,
        "reason": reason,
    }
    _logger.info(json.dumps(entry))

    _log_buffer.append(entry)
    if len(_log_buffer) > _MAX_BUFFER:
        _log_buffer.pop(0)

    return entry


def get_recent(n: int = 100) -> list[dict]:
    """Return the most recent *n* log entries (newest first)."""
    return list(reversed(_log_buffer[-n:]))
