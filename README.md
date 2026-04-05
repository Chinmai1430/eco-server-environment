---
title: Eco Server Env Environment Server
emoji: 🌍
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - pytorch
  - reinforcement-learning
  - sustainability
  - real-world
---

# 🌍 EcoServer Environment

> **An AI-powered ecological data center management environment for the Meta × Scaler OpenEnv Hackathon.**

A reinforcement learning environment where an AI agent manages a **15×15 server grid**,
making real-world decisions about energy usage, pollution control, server cooling,
and renewable energy deployment — all to maximize ecological sustainability.

---

## 🎯 Why EcoServer?

Data centers consume **1-2% of global electricity** and produce significant carbon emissions.
EcoServer simulates the real-world challenge of making data centers **greener and more efficient**
using AI — a problem with genuine environmental impact.

The agent must:
- 🌡️ **Cool overheating servers** before they cascade and cause outages
- 🌳 **Plant trees** and install solar panels to improve green cover
- 🧹 **Remove pollution** that spreads across the grid
- ⚡ **Upgrade server efficiency** to reduce energy consumption
- ⚖️ **Balance uptime vs sustainability** — you can't just shut everything down

---

## 🚀 Quick Start

### Connect to the Live Environment
```python
import requests

BASE_URL = "https://chinmai1430-eco-server-environment.hf.space"

# Reset environment
response = requests.post(f"{BASE_URL}/reset")
obs = response.json()
print(f"Eco Score: {obs['observation']['eco_score']}")
print(f"Pollution: {obs['observation']['pollution']}")

# Take actions
response = requests.post(f"{BASE_URL}/step", json={
    "action_type": "cool_server",
    "x": 7,
    "y": 7
})
result = response.json()
print(f"Reward: {result['reward']}")
print(f"Events: {result['info']['reward_events']}")
```

### Run Locally
```bash
git clone https://github.com/Chinmai1430/eco-server-environment
cd eco-server-environment
pip install -r requirements.txt
python server/app.py
```

### Docker
```bash
docker build -t eco-server .
docker run -p 7860:7860 eco-server
```

---

## 🧠 DQN AI Agent

EcoServer includes a **Deep Q-Network (DQN)** baseline agent built with PyTorch:
Input Layer  →  128 neurons (ReLU)
↓
Hidden Layer →  256 neurons (ReLU + Dropout)
↓
Hidden Layer →  128 neurons (ReLU)
↓
Output Layer →   8 Q-values (one per action)

### Agent Features
- ✅ **Experience Replay** — 10,000 sample buffer for stable training
- ✅ **Target Network** — Updated every 10 episodes for Q-value stability
- ✅ **Epsilon-Greedy** — Exploration decays from 1.0 → 0.05
- ✅ **Huber Loss** — Robust to reward outliers
- ✅ **Gradient Clipping** — Prevents exploding gradients
- ✅ **Xavier Initialization** — Better weight initialization
- ✅ **Rule-Based Fallback** — Works even without PyTorch

### Train the Agent
```bash
python inference.py --mode train --episodes 200
```

### Evaluate the Agent
```bash
python inference.py --mode eval
```

---

## 🗺️ Environment Details

### Grid Cell Types
| Cell | Symbol | Description |
|------|--------|-------------|
| Empty | ⬜ | Available space |
| Server | 🖥️ | Normal server rack |
| Hot Server | 🔥 | Overheating server (bad!) |
| Efficient Server | ⚡ | Upgraded low-energy server |
| Tree | 🌳 | Green cover, cleans pollution |
| Solar Panel | ☀️ | Renewable energy source |
| Polluted | ☠️ | Polluted cell, spreads slowly |

### Environment Dynamics (Realistic!)
Every step, the environment evolves:
- 🔥 Hot servers **spread heat** to neighboring servers (5% chance)
- 🖥️ Normal servers **randomly overheat** under load (2% chance)
- ☠️ Pollution **spreads** to empty neighbors (3% chance)
- 🌳 Trees **naturally clean** neighboring pollution (8% chance)

---

## ⚡ Action Space

| Action | Effect | Reward Bonus |
|--------|--------|-------------|
| `plant_tree` | Plants tree at (x,y) | +0.15 |
| `remove_pollution` | Cleans pollution + neighbors | +0.20 |
| `install_solar` | Adds renewable energy | +0.18 |
| `cool_server` | Fixes overheating server | +0.15 |
| `upgrade_efficiency` | Reduces server energy use | +0.12 |
| `decommission` | Shuts down overloaded server | +0.05 |
| `monitor` | Observe without side effects | +0.03 |
| `develop` | Adds server (increases load) | -0.15 ⚠️ |

---

## 📊 Observation Space
```python
{
  "grid":            [[int]],  # 15×15 grid of cell types
  "eco_score":       float,    # Overall health (0.0–1.0)
  "pollution":       float,    # Pollution level (lower = better)
  "green_cover":     float,    # % trees + solar panels
  "temperature":     float,    # Server heat level (lower = better)
  "energy_usage":    float,    # Total energy consumption
  "renewable_ratio": float,    # % energy from renewables
  "server_load":     float,    # % of grid running servers
  "uptime":          float,    # Server availability (higher = better)
  "step":            int,      # Current step number
  "done":            bool,     # Episode ended?
  "reward":          float,    # Last step reward
  "last_event":      str       # Human-readable action result
}
```

---

## 🏆 Reward Function

A **multi-signal reward** with 8 components:
reward = eco_score     × 0.25   # Current ecological health
+ Δ_improvement × 0.20   # Progress over last step
+ (1-pollution)  × 0.15  # Pollution reduction
+ renewable      × 0.15  # Renewable energy ratio
+ (1-temperature)× 0.10  # Server cooling
+ uptime         × 0.10  # Server availability
+ action_bonus   × 0.05  # Per-action shaping
+ milestone_bonus         # One-time achievement bonuses
+ time_efficiency          # Finish early bonus
- penalties                # Heat / pollution / low uptime

### Milestone Bonuses
| Achievement | Bonus |
|-------------|-------|
| 🌱 Eco score > 50% | +0.05 |
| 🌿 Eco score > 70% | +0.08 |
| 🌳 Eco score > 85% | +0.12 |
| 🏆 Eco score > 95% | +0.20 |
| ✨ Pollution < 30% | +0.06 |
| 💎 Pollution < 10% | +0.10 |
| ☀️ 50% renewable energy | +0.08 |
| ⚡ 80% renewable energy | +0.12 |

---

## 📋 Tasks

### Task 1 — Easy: Basic Environmental Monitoring
> Get familiar with the environment through observation

- **Goal**: Use `monitor` 5+ times, keep eco_score above 40%
- **Max Steps**: 10
- **Grading**: `monitor_steps/5 × 50% + eco_progress × 50%`
- **Baseline Score**: `0.90`
```python
# Example solution
for _ in range(5):
    requests.post(f"{BASE_URL}/step", json={"action_type": "monitor"})
```

---

### Task 2 — Medium: Pollution & Temperature Control
> Clean up the environment and cool down servers

- **Goal**: Pollution < 30%, Temperature < 20%, Eco score > 50%
- **Max Steps**: 10
- **Grading**: `pollution_score × 35% + temp_score × 30% + eco_score × 35%`
- **Baseline Score**: `0.65`
```python
# Example solution
actions = [
    {"action_type": "cool_server",        "x": 3, "y": 3},
    {"action_type": "remove_pollution",   "x": 5, "y": 5},
    {"action_type": "cool_server",        "x": 8, "y": 8},
    {"action_type": "remove_pollution",   "x": 2, "y": 9},
    {"action_type": "plant_tree",         "x": 7, "y": 7},
]
for action in actions:
    requests.post(f"{BASE_URL}/step", json=action)
```

---

### Task 3 — Hard: Full Ecosystem Recovery
> Transform a degraded data center into a sustainable one

- **Goal**: Eco score ≥ 85%, Renewable ratio ≥ 30%, within 20 steps
- **Max Steps**: 20
- **Grading**: `eco_progress × 70% + renewable_progress × 30%` + efficiency bonus
- **Baseline Score**: `0.42`
```python
# Example DQN agent solution
agent = DQNAgent()
agent.load("dqn_ecoserver_best.pth")

obs = requests.post(f"{BASE_URL}/reset").json()
for step in range(20):
    state = obs_to_state(obs["observation"])
    action_idx, x, y = agent.select_action(state)
    obs = requests.post(f"{BASE_URL}/step", json={
        "action_type": ACTIONS[action_idx],
        "x": x, "y": y
    }).json()
    if obs["done"]:
        break
```

---

## 📈 Baseline Scores

| Task | Random Agent | Rule-Based | DQN Agent |
|------|-------------|------------|-----------|
| task_easy | 0.72 | 0.88 | **0.90** |
| task_medium | 0.31 | 0.58 | **0.65** |
| task_hard | 0.18 | 0.35 | **0.42** |

---

## 🌐 API Reference

### `POST /reset`
Reset environment to degraded starting state.
```json
// Response
{
  "status": "ok",
  "observation": { "eco_score": 0.31, "pollution": 0.42, ... },
  "message": "Environment reset successfully"
}
```

### `POST /step`
Execute an action.
```json
// Request
{ "action_type": "cool_server", "x": 7, "y": 7 }

// Response
{
  "observation": { "eco_score": 0.34, ... },
  "reward": 0.4821,
  "done": false,
  "info": {
    "reward_breakdown": { "eco": 0.085, "delta": 0.062, ... },
    "reward_events": ["📈 Eco score improved by 0.030", "❄️ Cooled server at (7,7)"],
    "task_grades": { "task_easy": 0.9, "task_medium": 0.61, "task_hard": 0.38 }
  }
}
```

### `GET /state`
Get current environment state.

### `GET /health`
Health check.

### `GET /docs`
Interactive Swagger API documentation.

---

## 🗂️ Project Structure
eco-server-environment/
├── 📄 Dockerfile                          # Container definition
├── 📄 README.md                           # This file
├── 📄 openenv.yaml                        # OpenEnv manifest
├── 📄 inference.py                        # PyTorch DQN agent
├── 📄 visualization.py                    # Grid visualization
├── 📄 requirements.txt                    # Python dependencies
├── 📄 pyproject.toml                      # Project metadata
└── 📁 server/
├── 📄 app.py                          # FastAPI + reward engine
└── 📄 eco_server_env_environment.py   # Core environment

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Environment | Python 3.11 |
| API Server | FastAPI + Uvicorn |
| AI Agent | PyTorch (DQN) |
| Deployment | HuggingFace Spaces + Docker |
| Grid Size | 15×15 = 225 cells |
| Actions | 8 real-world actions |
| Observations | 8 ecological metrics |

---

## 📬 Contact & Links

- 🤗 **HuggingFace**: [Chinmai1430/eco-server-environment](https://huggingface.co/spaces/Chinmai1430/eco-server-environment)
- 💻 **GitHub**: [Chinmai1430/eco-server-environment](https://github.com/Chinmai1430/eco-server-environment)
- 📧 **Hackathon Help**: help_openenvhackathon@scaler.com

---

*Built for the **Meta × Scaler OpenEnv Hackathon** | Deadline: April 8th, 2026*
