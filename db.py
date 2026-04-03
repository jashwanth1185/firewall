"""
db.py  –  MongoDB integration for FireRL.

Collections
───────────
  packets      – every classified packet (capped at 100 000 docs)
  decisions    – every RL ALLOW/BLOCK decision with reward
  blocked_ips  – currently blocked IPs
  rule_changes – dynamic threshold adaptation history
  agent_stats  – periodic snapshots of Q-table and agent performance
"""

import os
import logging
import time
from datetime import datetime, timezone

from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure

log = logging.getLogger("firerl.db")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.environ.get("MONGO_DB",  "firerl")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FireRLDB:
    """Thin wrapper around PyMongo collections."""

    def __init__(self) -> None:
        self._client = None
        self._db     = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
            self._client.admin.command("ping")
            self._db = self._client[DB_NAME]
            self._ensure_indexes()
            self._connected = True
            log.info("Connected to MongoDB: %s / %s", MONGO_URI, DB_NAME)
            return True
        except (ConnectionFailure, OperationFailure) as e:
            log.warning("MongoDB unavailable (%s). Running without persistence.", e)
            self._connected = False
            return False

    def _ensure_indexes(self) -> None:
        db = self._db
        db.packets.create_index([("timestamp", DESCENDING)])
        db.packets.create_index([("src_ip", DESCENDING)])
        db.packets.create_index([("attack_type", DESCENDING)])
        db.decisions.create_index([("timestamp", DESCENDING)])
        db.decisions.create_index([("src_ip", DESCENDING)])
        db.blocked_ips.create_index([("ip", 1)], unique=True)
        db.rule_changes.create_index([("timestamp", DESCENDING)])
        db.agent_stats.create_index([("timestamp", DESCENDING)])
        log.info("MongoDB indexes ensured.")

    @property
    def ok(self) -> bool:
        return self._connected

    # ── packets ───────────────────────────────────────────────────────────────

    def insert_packet(self, record: dict) -> None:
        if not self._connected:
            return
        try:
            doc = {**record, "created_at": _utcnow()}
            self._db.packets.insert_one(doc)
        except Exception as e:
            log.debug("insert_packet error: %s", e)

    def get_packets(self, n: int = 100, attack_type: str | None = None) -> list[dict]:
        if not self._connected:
            return []
        try:
            flt = {}
            if attack_type:
                flt["attack_type"] = attack_type
            cursor = self._db.packets.find(flt, {"_id": 0}).sort(
                "timestamp", DESCENDING).limit(n)
            return list(cursor)
        except Exception as e:
            log.debug("get_packets error: %s", e)
            return []

    def get_packet_stats(self) -> dict:
        if not self._connected:
            return {}
        try:
            pipeline = [
                {"$group": {
                    "_id": "$attack_type",
                    "count": {"$sum": 1},
                    "avg_threat": {"$avg": "$threat_score"},
                }},
                {"$sort": {"count": -1}},
            ]
            rows = list(self._db.packets.aggregate(pipeline))
            return {r["_id"]: {"count": r["count"], "avg_threat": round(r["avg_threat"], 3)}
                    for r in rows}
        except Exception as e:
            log.debug("get_packet_stats error: %s", e)
            return {}

    # ── decisions ─────────────────────────────────────────────────────────────

    def insert_decision(self, record: dict) -> None:
        if not self._connected:
            return
        try:
            self._db.decisions.insert_one({**record, "created_at": _utcnow()})
        except Exception as e:
            log.debug("insert_decision error: %s", e)

    def get_decisions(self, n: int = 100) -> list[dict]:
        if not self._connected:
            return []
        try:
            return list(self._db.decisions.find(
                {}, {"_id": 0}).sort("timestamp", DESCENDING).limit(n))
        except Exception as e:
            log.debug("get_decisions error: %s", e)
            return []

    def decision_summary(self) -> dict:
        if not self._connected:
            return {}
        try:
            pipeline = [
                {"$group": {
                    "_id": "$action",
                    "count": {"$sum": 1},
                    "avg_reward": {"$avg": "$reward"},
                }}
            ]
            rows = list(self._db.decisions.aggregate(pipeline))
            return {r["_id"]: {"count": r["count"], "avg_reward": round(r["avg_reward"], 3)}
                    for r in rows}
        except Exception as e:
            log.debug("decision_summary error: %s", e)
            return {}

    # ── blocked IPs ───────────────────────────────────────────────────────────

    def upsert_blocked(self, ip: str, meta: dict) -> None:
        if not self._connected:
            return
        try:
            self._db.blocked_ips.update_one(
                {"ip": ip},
                {"$set": {**meta, "ip": ip, "updated_at": _utcnow()}},
                upsert=True,
            )
        except Exception as e:
            log.debug("upsert_blocked error: %s", e)

    def remove_blocked(self, ip: str) -> None:
        if not self._connected:
            return
        try:
            self._db.blocked_ips.delete_one({"ip": ip})
        except Exception as e:
            log.debug("remove_blocked error: %s", e)

    def get_blocked(self) -> list[dict]:
        if not self._connected:
            return []
        try:
            return list(self._db.blocked_ips.find({}, {"_id": 0}))
        except Exception as e:
            log.debug("get_blocked error: %s", e)
            return []

    # ── rule changes ──────────────────────────────────────────────────────────

    def insert_rule_change(self, snapshot: dict) -> None:
        if not self._connected:
            return
        try:
            self._db.rule_changes.insert_one({**snapshot, "timestamp": _utcnow()})
        except Exception as e:
            log.debug("insert_rule_change error: %s", e)

    def get_rule_changes(self, n: int = 50) -> list[dict]:
        if not self._connected:
            return []
        try:
            return list(self._db.rule_changes.find(
                {}, {"_id": 0}).sort("timestamp", DESCENDING).limit(n))
        except Exception as e:
            log.debug("get_rule_changes error: %s", e)
            return []

    # ── agent snapshots ───────────────────────────────────────────────────────

    def insert_agent_snapshot(self, stats: dict) -> None:
        if not self._connected:
            return
        try:
            self._db.agent_stats.insert_one({**stats, "timestamp": _utcnow()})
        except Exception as e:
            log.debug("insert_agent_snapshot error: %s", e)

    def get_agent_history(self, n: int = 200) -> list[dict]:
        if not self._connected:
            return []
        try:
            return list(self._db.agent_stats.find(
                {}, {"_id": 0, "q_table_sample": 0}).sort(
                "timestamp", DESCENDING).limit(n))
        except Exception as e:
            log.debug("get_agent_history error: %s", e)
            return []


# Singleton
db = FireRLDB()
