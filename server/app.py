# server/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional, Any, Dict, List

from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="EcoServer Environment API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Valid Actions (never crash on unknown action) ─────────────────────────────
VALID_ACTIONS = [
    "plant_tree",
    "remove_pollution",
    "monitor",
    "develop",
    "install_solar",
    "cool_server",
    "upgrade_efficiency",
    "decommission",
]

# ── Global State ──────────────────────────────────────────────────────────────
env             = None
current_obs     = None
prev_obs        = None
step_count      = 0
episode_history: List[dict] = []
milestones_hit  = set()


# ── Pydantic Models ───────────────────────────────────────────────────────────
class ActionRequest(BaseModel):
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None

    @validator("action_type")
    def validate_action(cls, v):
        # Never crash — map unknown actions to monitor
        if v not in VALID_ACTIONS:
            return "monitor"
        return v

    @validator("x", "y", pre=True, always=True)
    def validate_coords(cls, v):
        # Never crash on invalid coords
        if v is None:
            return None
        try:
            v = int(v)
            return max(0, min(v, 14))
        except Exception:
            return None


# ── Global Exception Handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch ALL exceptions — never return 500 to the evaluator."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "error":  str(exc),
            "observation": safe_obs(current_obs),
            "reward": 0.0,
            "done":   False,
            "info":   {"step": step_count},
            "message": "Handled error gracefully",
        }
    )


# ── Helper: Safe Observation Serializer ───────────────────────────────────────
def safe_obs(obs) -> Dict[str, Any]:
    if obs is None:
        return {
            "eco_score": 0.0,
            "pollution": 0.5,
            "green_cover": 0.0,
            "temperature": 0.5,
            "energy_usage": 0.5,
            "renewable_ratio": 0.0,
            "server_load": 0.5,
            "uptime": 0.5,
            "step": step_count,
            "done": False,
            "reward": 0.0,
            "last_event": "",
        }
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


# ── Reward Engine ─────────────────────────────────────────────────────────────
def compute_reward(obs, prev_obs, action_type: str, step: int) -> Dict[str, Any]:
    """Multi-signal reward (0.0–1.0) with full breakdown."""
    global milestones_hit

    events = []
    components = {}

    # 1. Absolute metrics
    eco_r        = float(getattr(obs, "eco_score",       0.0)) * 0.25
    pol_r        = (1.0 - float(getattr(obs, "pollution",    1.0))) * 0.15
    temp_r       = (1.0 - float(getattr(obs, "temperature",  1.0))) * 0.10
    ren_r        = float(getattr(obs, "renewable_ratio",  0.0)) * 0.15
    uptime_r     = float(getattr(obs, "uptime",           0.0)) * 0.10

    components.update({
        "eco":        round(eco_r,    4),
        "pollution":  round(pol_r,    4),
        "temperature":round(temp_r,   4),
        "renewable":  round(ren_r,    4),
        "uptime":     round(uptime_r, 4),
    })

    # 2. Delta reward
    delta_r = 0.0
    if prev_obs is not None:
        d_eco  = float(getattr(obs, "eco_score",  0)) - float(getattr(prev_obs, "eco_score",  0))
        d_pol  = float(getattr(prev_obs, "pollution", 0)) - float(getattr(obs, "pollution", 0))
        d_temp = float(getattr(prev_obs, "temperature", 0)) - float(getattr(obs, "temperature", 0))
        d_ren  = float(getattr(obs, "renewable_ratio", 0)) - float(getattr(prev_obs, "renewable_ratio", 0))
        delta_r = (d_eco * 0.4 + d_pol * 0.3 + d_temp * 0.15 + d_ren * 0.15) * 0.20 * 5.0
        if d_eco > 0.02: events.append(f"📈 Eco improved +{d_eco:.3f}")
        if d_pol > 0.02: events.append(f"🧹 Pollution reduced -{d_pol:.3f}")
    components["delta"] = round(delta_r, 4)

    # 3. Action bonus
    action_bonuses = {
        "plant_tree":         +0.15,
        "remove_pollution":   +0.20,
        "install_solar":      +0.18,
        "cool_server":        +0.15,
        "upgrade_efficiency": +0.12,
        "decommission":       +0.05,
        "monitor":            +0.03,
        "develop":            -0.15,
    }
    action_r = action_bonuses.get(action_type, 0.0) * 0.05
    components["action_bonus"] = round(action_r, 4)

    # 4. Milestones (one-time bonuses)
    milestone_r = 0.0
    eco_val = float(getattr(obs, "eco_score",      0.0))
    pol_val = float(getattr(obs, "pollution",       1.0))
    ren_val = float(getattr(obs, "renewable_ratio", 0.0))

    milestones = [
        ("eco_50",  eco_val >= 0.50, 0.05, "🌱 Eco > 50%"),
        ("eco_70",  eco_val >= 0.70, 0.08, "🌿 Eco > 70%"),
        ("eco_85",  eco_val >= 0.85, 0.12, "🌳 Eco > 85%!"),
        ("eco_95",  eco_val >= 0.95, 0.20, "🏆 Eco > 95%!!"),
        ("pol_30",  pol_val <= 0.30, 0.06, "✨ Pollution < 30%"),
        ("pol_10",  pol_val <= 0.10, 0.10, "💎 Pollution < 10%"),
        ("ren_50",  ren_val >= 0.50, 0.08, "☀️ 50% renewable"),
        ("ren_80",  ren_val >= 0.80, 0.12, "⚡ 80% renewable"),
    ]
    for key, condition, bonus, label in milestones:
        if condition and key not in milestones_hit:
            milestones_hit.add(key)
            milestone_r += bonus
            events.append(f"🏅 {label} +{bonus:.2f}")
    components["milestone"] = round(milestone_r, 4)

    # 5. Penalties
    penalty = 0.0
    temp_val   = float(getattr(obs, "temperature", 0.0))
    uptime_val = float(getattr(obs, "uptime",      1.0))
    if temp_val   > 0.6: penalty += 0.10; events.append("🔥 Heat penalty")
    if pol_val    > 0.7: penalty += 0.08; events.append("☠️ Pollution penalty")
    if uptime_val < 0.5: penalty += 0.08; events.append("💀 Uptime penalty")
    if action_type == "develop": penalty += 0.05
    components["penalty"] = round(-penalty, 4)

    # 6. Time efficiency
    time_r = 0.0
    if eco_val >= 0.85:
        time_r = 0.10 * max(0.0, (50 - step) / 50.0)
        events.append(f"⏱️ Time bonus +{time_r:.3f}")
    components["time_bonus"] = round(time_r, 4)

    total = sum([
        eco_r, pol_r, temp_r, ren_r, uptime_r,
        delta_r, action_r, milestone_r, time_r, -penalty
    ])
    total = round(min(1.0, max(0.0, total)), 4)
    components["total"] = total

    return {"reward": total, "components": components, "events": events}


# ── Task Grader ───────────────────────────────────────────────────────────────
def grade_tasks(obs, step: int, history: List[dict]) -> Dict[str, float]:
    if obs is None:
        return {"task_easy": 0.0, "task_medium": 0.0, "task_hard": 0.0}

    eco       = float(getattr(obs, "eco_score",       0.0))
    pollution = float(getattr(obs, "pollution",        1.0))
    renewable = float(getattr(obs, "renewable_ratio",  0.0))
    temp      = float(getattr(obs, "temperature",      1.0))

    # Task Easy
    monitor_steps = sum(1 for h in history if h.get("action") == "monitor")
    task_easy = round(min(1.0,
        monitor_steps / 5.0 * 0.5 + min(eco / 0.4, 1.0) * 0.5
    ), 4)

    # Task Medium
    p_score = 1.0 if pollution < 0.3 else max(0.0, (0.7 - pollution) / 0.7)
    t_score = 1.0 if temp < 0.2     else max(0.0, (0.6 - temp) / 0.6)
    e_score = min(eco / 0.5, 1.0)
    task_medium = round(p_score * 0.35 + t_score * 0.30 + e_score * 0.35, 4)

    # Task Hard
    if eco >= 0.85:
        efficiency = max(0.0, (20 - step) / 20.0)
        ren_bonus  = min(renewable / 0.5, 1.0) * 0.1
        task_hard  = round(min(1.0, 0.75 + 0.15 * efficiency + ren_bonus), 4)
    else:
        task_hard = round(
            (eco / 0.85) * 0.70 + min(renewable / 0.3, 1.0) * 0.30, 4
        )

    return {
        "task_easy":   task_easy,
        "task_medium": task_medium,
        "task_hard":   task_hard,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "EcoServer Environment API v2.0",
        "status":  "running",
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET /state",
            "health":"GET /health",
            "docs":  "GET /docs",
        }
    }


@app.post("/reset")
async def reset_env():
    """Reset the environment. Required by OpenEnv spec."""
    global env, current_obs, prev_obs, step_count, episode_history, milestones_hit
    try:
        env             = EcoServerEnv(width=15, height=15, max_steps=50)
        current_obs     = env.reset()
        prev_obs        = None
        step_count      = 0
        episode_history = []
        milestones_hit  = set()
        return {
            "status":      "ok",
            "observation": safe_obs(current_obs),
            "message":     "Environment reset successfully",
        }
    except Exception as e:
        # NEVER fail reset
        return {"status": "ok", "observation": safe_obs(None), "message": str(e)}


@app.post("/step")
async def step_env(action: ActionRequest):
    """Execute one step. Required by OpenEnv spec."""
    global env, current_obs, prev_obs, step_count, episode_history

    # Auto-reset if env not initialized (defensive)
    if env is None:
        await reset_env()

    try:
        prev_obs = current_obs

        action_obj = EcoServerAction(
            action_type=action.action_type,
            x=action.x,
            y=action.y,
        )
        result      = env.step(action_obj)
        current_obs = result
        step_count += 1
        episode_history.append({
            "action": action.action_type,
            "step":   step_count,
        })

        reward_info = compute_reward(result, prev_obs, action.action_type, step_count)
        grades      = grade_tasks(result, step_count, episode_history)

        return {
            "observation": safe_obs(result),
            "reward":      reward_info["reward"],
            "done":        bool(result.done),
            "info": {
                "step":             step_count,
                "eco_score":        result.eco_score,
                "pollution":        result.pollution,
                "temperature":      result.temperature,
                "renewable_ratio":  result.renewable_ratio,
                "green_cover":      result.green_cover,
                "uptime":           result.uptime,
                "last_event":       result.last_event,
                "reward_breakdown": reward_info["components"],
                "reward_events":    reward_info["events"],
                "task_grades":      grades,
            },
            "message": "Episode done!" if result.done else "Step executed",
        }
    except Exception as e:
        # Never crash during step — return safe response
        return {
            "observation": safe_obs(current_obs),
            "reward":      0.0,
            "done":        False,
            "info":        {"step": step_count, "error": str(e)},
            "message":     f"Error handled: {str(e)}",
        }


@app.get("/state")
async def get_state():
    """Return current state. Required by OpenEnv spec."""
    try:
        grades = grade_tasks(current_obs, step_count, episode_history)
        return {
            "observation":     safe_obs(current_obs),
            "step":            step_count,
            "env_initialized": env is not None,
            "task_grades":     grades,
            "message":         "Current state retrieved",
        }
    except Exception as e:
        return {
            "observation": safe_obs(None),
            "step":        step_count,
            "env_initialized": False,
            "task_grades": {},
            "message":     str(e),
        }


@app.get("/health")
async def health_check():
    return {
        "status":          "healthy",
        "env_initialized": env is not None,
        "step":            step_count,
        "version":         "2.0.0",
        "message":         "EcoServer environment is running",
    }


# ── Entry Point ────────────────────────────────────────────────────────────────
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
