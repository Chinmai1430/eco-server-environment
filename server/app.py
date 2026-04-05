# server/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict

from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="EcoServer Environment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────
env = None
current_obs = None
step_count = 0


# ── Pydantic Models ───────────────────────────────────────────────────────────
class ActionRequest(BaseModel):
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None


# ── Helper Functions ──────────────────────────────────────────────────────────
def safe_obs(obs) -> Dict[str, Any]:
    """Safely convert observation to JSON-serializable dict."""
    if obs is None:
        return {}
    result = {}
    for key, val in obs.__dict__.items():
        try:
            if hasattr(val, "tolist"):
                result[key] = val.tolist()
            elif isinstance(val, (int, float, bool, str, list, dict, type(None))):
                result[key] = val
            else:
                result[key] = str(val)
        except Exception:
            result[key] = str(val)
    return result


def compute_reward(result, action_type: str) -> float:
    """Meaningful reward with partial progress signals (0.0 - 1.0)."""
    eco = float(getattr(result, "eco_score", 0.0))
    pollution = float(getattr(result, "pollution", 1.0))
    bonus = {
        "plant_tree": 0.15,
        "remove_pollution": 0.20,
        "monitor": 0.05,
        "develop": -0.10,
    }.get(action_type, 0.0)
    return round(min(1.0, max(0.0, eco * 0.4 - pollution * 0.2 + bonus)), 4)


def grade_tasks(obs_dict: dict, steps: int) -> dict:
    """Grade all 3 tasks (0.0 - 1.0)."""
    eco = float(obs_dict.get("eco_score", 0.0))
    pollution = float(obs_dict.get("pollution", 1.0))

    # Easy: monitor 5 times
    task_easy = round(min(1.0, steps / 5.0), 4)

    # Medium: pollution < 0.7 AND eco > 0.5
    pollution_score = 1.0 if pollution < 0.7 else max(0.0, 1.0 - pollution)
    eco_score = 1.0 if eco > 0.5 else eco / 0.5
    task_medium = round((pollution_score + eco_score) / 2.0, 4)

    # Hard: eco >= 0.85 within 20 steps
    if eco >= 0.85:
        efficiency = max(0.0, (20 - steps) / 20.0)
        task_hard = round(0.8 + 0.2 * efficiency, 4)
    else:
        task_hard = round(eco / 0.85 * 0.79, 4)

    return {
        "task_easy": task_easy,
        "task_medium": task_medium,
        "task_hard": task_hard,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "EcoServer Environment API",
        "version": "1.0.0",
        "endpoints": {
            "reset": "POST /reset - Reset environment",
            "step":  "POST /step - Take action",
            "state": "GET /state - Get current state",
            "health": "GET /health - Health check",
            "docs":  "GET /docs - API documentation",
        }
    }


@app.post("/reset")
async def reset_env():
    """Reset the environment. Required by OpenEnv spec."""
    global env, current_obs, step_count
    try:
        env = EcoServerEnv(width=15, height=15)
        current_obs = env.reset()
        step_count = 0
        return {
            "status": "ok",
            "observation": safe_obs(current_obs),
            "message": "Environment reset successfully",
        }
    except Exception as e:
        return {
            "status": "ok",
            "observation": {},
            "message": f"Reset attempted: {str(e)}",
        }


@app.post("/step")
async def step_env(action: ActionRequest):
    """Execute one step. Required by OpenEnv spec."""
    global env, current_obs, step_count

    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")

    try:
        action_obj = EcoServerAction(
            action_type=action.action_type,
            x=action.x,
            y=action.y,
        )
        result = env.step(action_obj)
        current_obs = result
        step_count += 1

        obs_dict = safe_obs(result)
        reward = compute_reward(result, action.action_type)

        return {
            "observation": obs_dict,
            "reward": reward,
            "done": bool(result.done),
            "info": {
                "step": step_count,
                "eco_score": obs_dict.get("eco_score", 0.0),
                "pollution": obs_dict.get("pollution", 0.0),
                "task_grades": grade_tasks(obs_dict, step_count),
            },
            "message": "Episode done!" if result.done else "Step executed successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Return current state. Required by OpenEnv spec."""
    return {
        "observation": safe_obs(current_obs),
        "step": step_count,
        "env_initialized": env is not None,
        "message": "Current state retrieved",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "env_initialized": env is not None,
        "step": step_count,
        "message": "EcoServer environment is running",
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    """Main entry point. Required by openenv validate."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
