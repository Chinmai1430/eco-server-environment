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
---

# EcoServer Environment

An AI-powered ecological server management environment for the Meta x Scaler OpenEnv Hackathon.
The agent manages a 15×15 server grid, balancing energy usage, pollution control,
and infrastructure development to maximize ecological sustainability.

## Quick Start
```python
from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

# Create and reset environment
env = EcoServerEnv(width=15, height=15)
obs = env.reset()

# Take actions
result = env.step(EcoServerAction(action_type="plant_tree", x=7, y=7))
print(f"Eco Score: {result.eco_score}")
print(f"Reward: {result.reward}")
```

## API Endpoints

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| POST   | /reset    | Reset environment to initial state |
| POST   | /step     | Execute an action                  |
| GET    | /state    | Get current environment state      |
| GET    | /health   | Health check                       |
| GET    | /docs     | Interactive API documentation      |

## Action Space

**ActionRequest**: Contains the action to perform
- `action_type` (str) - One of: `plant_tree`, `remove_pollution`, `monitor`, `develop`
- `x` (int, optional) - X coordinate on the 15×15 grid (0–14)
- `y` (int, optional) - Y coordinate on the 15×15 grid (0–14)

### Action Effects
| Action             | Effect                                      | Reward Bonus |
|--------------------|---------------------------------------------|--------------|
| `plant_tree`       | Plants a tree at (x,y), improves eco score  | +0.15        |
| `remove_pollution` | Cleans pollution at (x,y)                   | +0.20        |
| `monitor`          | Observes current state without side effects | +0.05        |
| `develop`          | Develops infrastructure, reduces eco score  | -0.10        |

## Observation Space

**ObservationResponse**: Contains the environment state
- `grid` (list) - 15×15 grid representing the ecosystem
- `eco_score` (float) - Ecological health score (0.0–1.0, higher is better)
- `pollution` (float) - Pollution level (0.0–1.0, lower is better)
- `step` (int) - Current step number
- `done` (bool) - Whether the episode has ended
- `reward` (float) - Reward for the last action (0.0–1.0)

## Reward Function

Reward is computed as:
reward = eco_score × 0.4 - pollution × 0.2 + action_bonus

Clipped to range [0.0, 1.0] with partial progress signals throughout the episode.

## Tasks

### Task 1 — Easy: Basic Environmental Monitoring
- **Goal**: Use `monitor` action 5 times without destructive actions
- **Max Steps**: 10
- **Baseline Score**: 0.90

### Task 2 — Medium: Pollution Control
- **Goal**: Reduce pollution below 0.7 while maintaining eco_score above 0.5
- **Max Steps**: 10
- **Baseline Score**: 0.65

### Task 3 — Hard: Full Ecosystem Recovery
- **Goal**: Achieve eco_score ≥ 0.85 within 20 steps
- **Max Steps**: 20
- **Baseline Score**: 0.42

## Baseline Scores

| Task        | Score |
|-------------|-------|
| task_easy   | 0.90  |
| task_medium | 0.65  |
| task_hard   | 0.42  |

## Setup & Running

### Using Docker
```bash
# Build
docker build -t eco-server .

# Run
docker run -p 7860:7860 eco-server
```

### Running Locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Connecting to the Deployed Space
```python
import requests

BASE_URL = "https://chinmai1430-eco-server-environment.hf.space"

# Reset
requests.post(f"{BASE_URL}/reset")

# Step
response = requests.post(f"{BASE_URL}/step", json={
    "action_type": "plant_tree",
    "x": 7,
    "y": 7
})
print(response.json())
```

## Project Structure

eco-server-environment/
├── Dockerfile                        # Container definition
├── README.md                         # This file
├── openenv.yaml                      # OpenEnv manifest
├── inference.py                      # Baseline inference script
├── visualization.py                  # Grid visualization utilities
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project metadata
└── server/
├── app.py                        # FastAPI application
└── eco_server_env_environment.py # Core environment logic

## Environment Details

- **Grid Size**: 15×15
- **Max Steps per Episode**: 20
- **Framework**: FastAPI + Uvicorn
- **Python**: 3.11
- **Port**: 7860

## Hugging Face Space

Deployed at:
`https://huggingface.co/spaces/Chinmai1430/eco-server-environment`

- **Web UI**: `/docs` — Interactive Swagger interface
- **Health Check**: `/health`
- **API**: `/reset`, `/step`, `/state`

