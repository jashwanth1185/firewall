# FireRL — Adaptive Firewall Intelligence

Full-stack reinforcement learning firewall with Flask backend, MongoDB persistence, and React dashboard.

## Architecture

```
firerl/
├── backend/
│   ├── app.py              ← Flask REST API (9 endpoints)
│   ├── classifier.py       ← 5-rule attack classification engine
│   ├── rl_agent.py         ← Q-Learning agent + dynamic rule adaptation
│   ├── packet_sniffer.py   ← Scapy sniffer + demo mode (no root required)
│   ├── firewall.py         ← Windows netsh / Linux iptables enforcement
│   ├── db.py               ← MongoDB integration (pymongo)
│   └── requirements.txt
└── frontend/
    └── index.html          ← React dashboard (CDN, no build step)
```

## How the RL Works

### State Space (243 states = 3^5)
| Dimension | Buckets |
|-----------|---------|
| Threat level | 0=Normal, 1=Suspicious, 2=Attack |
| Packet rate | 0=Low(<10pps), 1=Med(10-50), 2=High(>50) |
| Port diversity | 0=Few(<5), 1=Mod(5-15), 2=Many(>15) |
| Byte rate | 0=Low(<1KB/s), 1=Med(1K-100K), 2=High(>100K) |
| Protocol risk | 0=Safe(HTTP/S), 1=Neutral, 2=Risky(SSH/RDP/SMB) |

### Action Space
| Action | Meaning | Firewall |
|--------|---------|----------|
| 0 | ALLOW | Forward packet |
| 1 | BLOCK_TEMP | 5-minute OS-level block |
| 2 | BLOCK_PERM | Permanent block |

### Reward Shaping (prevents always-block policy)
| Outcome | Reward | Reason |
|---------|--------|--------|
| TP (attack blocked) | +10 | Correct action |
| TN (normal allowed) | +2 | Correct action |
| FP (normal blocked) | -8 | Penalise over-blocking |
| FN (attack allowed) | -6 | Penalise under-blocking |
| PERM correct block | +15 | Strong reward for confident correct blocks |
| PERM false block | -15 | Heavy penalty for confident mistakes |

Math: always-BLOCK EV = 0.3×10 + 0.7×(-8) = -2.6 (bad).
Correct-policy EV = 0.3×10 + 0.7×2 = +4.4 (best).

### Dynamic Rule Adaptation
Every 200 decisions the agent inspects its Q-table and adjusts classifier thresholds:
- If Q(attack_state, BLOCK) >> Q(attack_state, ALLOW) AND FP rate < 15%:
  → **Tighten** thresholds (catch more attacks)
- If FP rate > 20%:
  → **Loosen** thresholds (reduce false positives)
- If FN rate > 30%:
  → **Tighten** thresholds (stop missing attacks)

### Classification Rules (5-Rule Priority Chain)

| Priority | Type | Threshold | Action |
|----------|------|-----------|--------|
| 1 | DoS/DDoS | pps > 1000 OR bps > 10MB/s | DROP |
| 2 | Port Scan | unique_ports/10s > 15 AND pps < 50 | DENY |
| 3 | Brute Force | failed_auth/60s > 10 on auth ports | RESET-BOTH |
| 4 | Suspicious | bpp < 40B OR evil_port OR SYN-flood | DENY (LOW) |
| 5 | Normal | default permit | ALLOW |

## Setup

### 1. MongoDB
```bash
# Install MongoDB Community: https://www.mongodb.com/try/download/community
# Start service
mongod --dbpath /data/db
```

### 2. Python Backend
```bash
cd backend
pip install -r requirements.txt

# Windows (requires Npcap from https://npcap.com/ + Admin terminal):
python app.py

# Linux (requires root for packet capture):
sudo python app.py

# Demo mode (no root / no Npcap — synthetic traffic generator):
FIRERL_DEMO=1 python app.py

# Custom MongoDB URI or NIC:
MONGO_URI=mongodb://user:pass@host:27017/ FIRERL_IFACE=eth0 python app.py
```

### 3. Frontend
Open `frontend/index.html` directly in a browser.
Or serve it:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check, DB status, agent stats |
| GET | `/api/traffic?n=100` | Recent packets (live + DB) |
| GET | `/api/logs?n=100` | Decision log from MongoDB |
| GET | `/api/blocked` | Currently blocked IPs |
| GET | `/api/model` | Q-table, agent stats, DB history |
| GET | `/api/stats` | Aggregate packet/decision stats |
| GET | `/api/rule-history` | Dynamic threshold adaptation log |
| POST | `/api/decision` | Manual ALLOW/BLOCK_TEMP/BLOCK_PERM |
| POST | `/api/rl-mode` | Toggle RL auto-mode |
| POST | `/api/unblock` | Unblock an IP |
| POST | `/api/reset-agent` | Reset Q-table to zero |

## MongoDB Collections

| Collection | Contents |
|------------|----------|
| `packets` | Every classified packet with threat score, state, action |
| `decisions` | Every RL decision with reward (links to Q-update) |
| `blocked_ips` | Currently blocked IPs with expiry |
| `rule_changes` | Dynamic threshold adaptation history |
| `agent_stats` | Periodic Q-table snapshots (every 60s) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://localhost:27017/` | MongoDB connection string |
| `MONGO_DB` | `firerl` | Database name |
| `FIRERL_IFACE` | auto-detect | Network interface to sniff |
| `FIRERL_DEMO` | `0` | Set `1` to use synthetic traffic (no root needed) |
