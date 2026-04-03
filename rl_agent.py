"""
rl_agent.py  –  Q-Learning agent with dynamic rule threshold adaptation.

State space (5 dimensions → 3^5 = 243 states)
──────────────────────────────────────────────
  threat_level        : 0=normal | 1=suspicious | 2=attack
  packet_rate_bucket  : 0=low(<10) | 1=med(10-50) | 2=high(>50) pps
  port_diversity      : 0=few(<5) | 1=mod(5-15) | 2=many(>15)
  byte_rate_bucket    : 0=low(<1KB/s) | 1=med(1KB-100KB/s) | 2=high(>100KB/s)
  protocol_risk       : 0=safe(HTTP/HTTPS) | 1=neutral | 2=risky(SSH/RDP/SMB)

Action space (3 actions)
────────────────────────
  0 = ALLOW
  1 = BLOCK_TEMP  (5-minute block)
  2 = BLOCK_PERM  (permanent block)

Reward shaping  (shapes always-block EV < correct-policy EV)
─────────────────────────────────────────────────────────────
  TP  correct block (attack)    → +10
  TN  correct allow (normal)    →  +2
  FP  false positive (block OK) →  -8   (costly – prevents all-block policy)
  FN  false negative (allow bad)→  -6
  PERM correct block            → +15   (reward permanent blocks of real attacks)
  PERM false block              → -15   (heavily penalise permanent false blocks)

Dynamic rule adaptation
────────────────────────
  After every N_ADAPT_STEPS decisions the agent inspects its Q-table and
  updates the classifier thresholds (DOS_PPS, SCAN_PORTS, BRUTE_FAILED)
  based on the learned Q-values:
    – If Q(attack-state, BLOCK) >> Q(attack-state, ALLOW): tighten thresholds
    – If FP rate is rising: loosen thresholds
  This makes the rule engine evolve automatically with real traffic.

Q-update (Bellman)
──────────────────
  Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
"""

import json
import os
import threading
import logging
import numpy as np

log = logging.getLogger("firerl.agent")

# ── hyper-parameters ──────────────────────────────────────────────────────────
ALPHA           = 0.15     # learning rate
GAMMA           = 0.90     # discount factor
EPSILON_START   = 0.40     # initial exploration
EPSILON_MIN     = 0.05     # floor
EPSILON_DECAY   = 0.997    # per-decision decay
N_ADAPT_STEPS   = 200      # how often to adapt rules (decisions)

# State dims and action count
STATE_DIMS  = (3, 3, 3, 3, 3)   # 243 states
N_ACTIONS   = 3
ACTION_NAMES = {0: "ALLOW", 1: "BLOCK_TEMP", 2: "BLOCK_PERM"}

QTABLE_PATH = os.path.join(os.path.dirname(__file__), "qtable.json")


class QLearningAgent:
    """
    Thread-safe tabular Q-learning agent with dynamic rule adaptation.

    The Q-table (243 × 3) is persisted to disk and loaded on startup so the
    agent continues learning across restarts.
    """

    def __init__(self) -> None:
        self._lock           = threading.Lock()
        self.epsilon         = EPSILON_START
        self.total_decisions = 0
        self.correct         = 0
        self.fp_count        = 0
        self.fn_count        = 0
        self.tp_count        = 0
        self.tn_count        = 0
        self.rewards_history: list[float] = []
        self.rule_history:    list[dict]  = []  # tracks threshold changes over time
        self.q_table: np.ndarray = self._load_or_init()

        # Current adaptive thresholds (start from classifier defaults)
        import classifier as clf
        self.dos_pps_threshold    = clf.DOS_PPS_THRESHOLD
        self.scan_ports_threshold = clf.SCAN_PORTS_THRESHOLD
        self.brute_threshold      = clf.BRUTE_FAILED_THRESHOLD

    # ── Q-table persistence ───────────────────────────────────────────────────

    def _load_or_init(self) -> np.ndarray:
        if os.path.exists(QTABLE_PATH):
            try:
                with open(QTABLE_PATH) as f:
                    data = json.load(f)
                arr = np.array(data, dtype=float)
                if arr.shape == (*STATE_DIMS, N_ACTIONS):
                    log.info("Loaded Q-table from %s", QTABLE_PATH)
                    return arr
            except Exception as e:
                log.warning("Q-table load failed (%s); starting fresh.", e)
        return np.zeros((*STATE_DIMS, N_ACTIONS), dtype=float)

    def save(self) -> None:
        with self._lock:
            with open(QTABLE_PATH, "w") as f:
                json.dump(self.q_table.tolist(), f)

    # ── state encoding ────────────────────────────────────────────────────────

    @staticmethod
    def encode_state(
        threat_score: float,
        pkt_rate: float,
        distinct_ports: int,
        byte_rate: float,
        dst_port: int,
    ) -> tuple[int, int, int, int, int]:
        """Map continuous observations → 5-dim discrete state tuple."""
        # Threat level
        if threat_score < 0.30:
            t = 0
        elif threat_score < 0.65:
            t = 1
        else:
            t = 2

        # Packet rate
        if pkt_rate < 10:
            r = 0
        elif pkt_rate < 50:
            r = 1
        else:
            r = 2

        # Port diversity
        if distinct_ports < 5:
            p = 0
        elif distinct_ports < 15:
            p = 1
        else:
            p = 2

        # Byte rate
        if byte_rate < 1_000:
            b = 0
        elif byte_rate < 100_000:
            b = 1
        else:
            b = 2

        # Protocol risk
        RISKY_PORTS = {22, 23, 3389, 445, 1433, 3306, 5900, 6379}
        SAFE_PORTS  = {80, 443, 8080, 8443, 8000}
        if dst_port in SAFE_PORTS:
            pr = 0
        elif dst_port in RISKY_PORTS:
            pr = 2
        else:
            pr = 1

        return (t, r, p, b, pr)

    # ── decision ──────────────────────────────────────────────────────────────

    def decide(self, state: tuple) -> int:
        """ε-greedy action selection. Returns 0, 1, or 2."""
        with self._lock:
            if np.random.random() < self.epsilon:
                return int(np.random.randint(N_ACTIONS))
            return int(np.argmax(self.q_table[state]))

    # ── reward computation ────────────────────────────────────────────────────

    def compute_reward(self, action: int, attack_type: str, threat_score: float) -> float:
        """
        Reward shaped so:
          always-BLOCK EV < 0  (because FP penalty -8 × 0.7 > TP reward +10 × 0.3)
          correct-policy EV > 0
          always-ALLOW EV < 0
        """
        is_bad = (attack_type != "normal") or (threat_score >= 0.5)

        if action == 0:   # ALLOW
            return 2.0 if not is_bad else -6.0
        elif action == 1: # BLOCK_TEMP
            return 10.0 if is_bad else -8.0
        else:             # BLOCK_PERM
            return 15.0 if is_bad else -15.0

    # ── Q-update ──────────────────────────────────────────────────────────────

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        attack_type: str,
        threat_score: float,
    ) -> None:
        """Apply one Bellman Q-update and decay epsilon."""
        with self._lock:
            cq  = self.q_table[state][action]
            mnq = float(np.max(self.q_table[next_state]))
            nq  = cq + ALPHA * (reward + GAMMA * mnq - cq)
            self.q_table[state][action] = nq

            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
            self.total_decisions += 1

            # Track confusion matrix
            is_bad = (attack_type != "normal") or (threat_score >= 0.5)
            blocks = action > 0
            if is_bad and blocks:
                self.tp_count += 1; self.correct += 1
            elif not is_bad and not blocks:
                self.tn_count += 1; self.correct += 1
            elif not is_bad and blocks:
                self.fp_count += 1
            else:
                self.fn_count += 1

            self.rewards_history.append(reward)
            if len(self.rewards_history) > 1000:
                self.rewards_history.pop(0)

        # Adapt rule thresholds periodically
        if self.total_decisions % N_ADAPT_STEPS == 0:
            self._adapt_rules()

        # Periodic save
        if self.total_decisions % 100 == 0:
            self.save()

    # ── dynamic rule adaptation ───────────────────────────────────────────────

    def _adapt_rules(self) -> None:
        """
        Inspect Q-values to dynamically tighten or loosen classifier thresholds.

        Logic:
        ──────
        • If Q(high-threat, BLOCK) >> Q(high-threat, ALLOW) by a large margin,
          the agent has learned that blocking high-threat states is very beneficial
          → tighten thresholds (lower them) so MORE traffic gets flagged as attack.

        • If FP rate is rising (fp/(fp+tn) > 20%), loosen thresholds
          → raise them so fewer benign packets get misclassified.

        • After adapting, write the new thresholds to the classifier module so
          live classification immediately uses the updated values.
        """
        import classifier as clf

        with self._lock:
            # Q-value for (attack, high-rate, many-ports, high-bps, risky) state
            attack_state = (2, 2, 2, 2, 2)
            q_block = float(np.max(self.q_table[attack_state][1:]))  # max of BLOCK actions
            q_allow = float(self.q_table[attack_state][0])
            block_preference = q_block - q_allow

            fp_rate = self.fp_count / max(self.fp_count + self.tn_count, 1)
            fn_rate = self.fn_count / max(self.fn_count + self.tp_count, 1)

            old_dos   = self.dos_pps_threshold
            old_scan  = self.scan_ports_threshold
            old_brute = self.brute_threshold
            changed   = False

            # Tighten: agent strongly prefers blocking attack states AND low FP rate
            if block_preference > 5.0 and fp_rate < 0.15:
                self.dos_pps_threshold    = max(500,  int(self.dos_pps_threshold    * 0.9))
                self.scan_ports_threshold = max(8,    int(self.scan_ports_threshold * 0.9))
                self.brute_threshold      = max(5,    int(self.brute_threshold      * 0.9))
                changed = True
                log.info("ADAPT ↓ tighten thresholds (block_pref=%.1f, fp=%.1f%%)",
                         block_preference, fp_rate * 100)

            # Loosen: FP rate too high (blocking too much legit traffic)
            elif fp_rate > 0.20:
                self.dos_pps_threshold    = min(2000, int(self.dos_pps_threshold    * 1.1))
                self.scan_ports_threshold = min(30,   int(self.scan_ports_threshold * 1.1))
                self.brute_threshold      = min(20,   int(self.brute_threshold      * 1.1))
                changed = True
                log.info("ADAPT ↑ loosen thresholds (fp=%.1f%%)", fp_rate * 100)

            # Tighten scan if FN rate is high (missing attacks)
            if fn_rate > 0.30:
                self.scan_ports_threshold = max(8, int(self.scan_ports_threshold * 0.85))
                self.dos_pps_threshold    = max(500, int(self.dos_pps_threshold  * 0.85))
                changed = True
                log.info("ADAPT ↓ fn_rate=%.1f%% – tightening", fn_rate * 100)

        # Push updated thresholds into the live classifier (no lock needed – Python GIL)
        if changed:
            clf.DOS_PPS_THRESHOLD    = self.dos_pps_threshold
            clf.SCAN_PORTS_THRESHOLD = self.scan_ports_threshold
            clf.BRUTE_FAILED_THRESHOLD = self.brute_threshold

            snapshot = {
                "step":              self.total_decisions,
                "dos_pps":           self.dos_pps_threshold,
                "scan_ports":        self.scan_ports_threshold,
                "brute_failed":      self.brute_threshold,
                "fp_rate":           round(fp_rate, 4),
                "fn_rate":           round(fn_rate, 4),
                "block_preference":  round(block_preference, 3),
            }
            self.rule_history.append(snapshot)
            if len(self.rule_history) > 200:
                self.rule_history.pop(0)

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            total = max(self.total_decisions, 1)
            acc   = self.correct / total
            avg_r = float(np.mean(self.rewards_history[-100:])) if self.rewards_history else 0.0
            return {
                "epsilon":             round(self.epsilon, 4),
                "total_decisions":     self.total_decisions,
                "correct_decisions":   self.correct,
                "tp":                  self.tp_count,
                "fp":                  self.fp_count,
                "tn":                  self.tn_count,
                "fn":                  self.fn_count,
                "accuracy":            round(acc, 4),
                "fp_rate":             round(self.fp_count / max(self.fp_count + self.tn_count, 1), 4),
                "fn_rate":             round(self.fn_count / max(self.fn_count + self.tp_count, 1), 4),
                "avg_reward_last_100": round(avg_r, 4),
                "q_table_shape":       list(self.q_table.shape),
                "q_table_sample":      self.q_table[:3, :3, :3, 0, 0].tolist(),  # compact slice
                "dynamic_thresholds": {
                    "dos_pps":     self.dos_pps_threshold,
                    "scan_ports":  self.scan_ports_threshold,
                    "brute_failed":self.brute_threshold,
                },
                "rule_history":        list(self.rule_history[-10:]),
            }


# Singleton
agent = QLearningAgent()
