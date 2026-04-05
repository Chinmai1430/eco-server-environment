# server/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="EcoServer Environment API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────
env          = None
current_obs  = None
prev_obs     = None   # ← used for delta-based rewards
step_count   = 0
episode_history: List[dict] = []


# ── Pydantic Models ───────────────────────────────────────────────────────────
class ActionRequest(BaseModel):
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None


# ── Helper: Safe Observation Serializer ───────────────────────────────────────
def safe_obs(obs) -> Dict[str, Any]:
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


# ── Reward Engine ─────────────────────────────────────────────────────────────
class RewardEngine:
    """
    Multi-signal reward function with:
    - Absolute metric rewards (current state quality)
    - Delta rewards (improvement over last step)
    - Action-specific shaping
    - Milestone bonuses
    - Penalty signals
    - Time efficiency bonus
    """

    # Weights for each reward component (must sum ≈ 1.0)
    W_ECO         = 0.25   # Overall eco score
    W_DELTA       = 0.20   # Improvement over last step
    W_POLLUTION   = 0.15   # Pollution reduction
    W_TEMP        = 0.10   # Temperature control
    W_RENEWABLE   = 0.15   # Renewable energy ratio
    W_UPTIME      = 0.10   # Server uptime
    W_ACTION      = 0.05   # Action-specific shaping

    # Action bonuses — rewards good actions, penalizes bad ones
    ACTION_BONUS = {
        "plant_tree":         +0.15,
        "remove_pollution":   +0.20,
        "install_solar":      +0.18,
        "cool_server":        +0.15,
        "upgrade_efficiency": +0.12,
        "monitor":            +0.03,
        "decommission":       +0.05,
        "develop":            -0.15,  # penalize — increases load & heat
    }

    # Milestone thresholds — bonus when crossed
    MILESTONES = [
        ("eco_score",       0.50, 0.05,  "🌱 Eco score above 50%"),
        ("eco_score",       0.70, 0.08,  "🌿 Eco score above 70%"),
        ("eco_score",       0.85, 0.12,  "🌳 Eco score above 85%!"),
        ("eco_score",       0.95, 0.20,  "🏆 Eco score above 95%!!"),
        ("pollution",       0.30, 0.06,  "✨ Pollution below 30%"),
        ("pollution",       0.10, 0.10,  "💎 Pollution below 10%"),
        ("renewable_ratio", 0.50, 0.08,  "☀️ 50% renewable energy"),
        ("renewable_ratio", 0.80, 0.12,  "⚡ 80% renewable energy"),
        ("temperature",     0.20, 0.06,  "❄️ Temperature under control"),
        ("uptime",          0.95, 0.05,  "💻 Excellent server uptime"),
    ]

    def __init__(self):
        self.milestones_hit = set()

    def reset(self):
        self.milestones_hit = set()

    def compute(
        self,
        obs,
        prev_obs,
        action_type: str,
        step: int,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Compute full reward breakdown.
        Returns dict with total reward and all components.
        """
        components = {}
        events     = []

        # ── 1. Absolute Metric Rewards ────────────────────────────────────────
        eco_reward        = obs.eco_score * self.W_ECO
        pollution_reward  = (1.0 - obs.pollution) * self.W_POLLUTION
        temp_reward       = (1.0 - obs.temperature) * self.W_TEMP
        renewable_reward  = obs.renewable_ratio * self.W_RENEWABLE
        uptime_reward     = obs.uptime * self.W_UPTIME

        components["eco"]        = round(eco_reward, 4)
        components["pollution"]  = round(pollution_reward, 4)
        components["temperature"]= round(temp_reward, 4)
        components["renewable"]  = round(renewable_reward, 4)
        components["uptime"]     = round(uptime_reward, 4)

        # ── 2. Delta Rewards (improvement over previous step) ─────────────────
        delta_reward = 0.0
        if prev_obs is not None:
            delta_eco   = obs.eco_score      - prev_obs.eco_score
            delta_pol   = prev_obs.pollution  - obs.pollution       # positive = improvement
            delta_temp  = prev_obs.temperature - obs.temperature
            delta_ren   = obs.renewable_ratio - prev_obs.renewable_ratio

            delta_reward = (
                delta_eco  * 0.40 +
                delta_pol  * 0.30 +
                delta_temp * 0.15 +
                delta_ren  * 0.15
            ) * self.W_DELTA * 5.0  # scale up delta signal

            if delta_eco > 0.02:
                events.append(f"📈 Eco score improved by {delta_eco:.3f}")
            if delta_pol > 0.02:
                events.append(f"🧹 Pollution reduced by {delta_pol:.3f}")

        components["delta"] = round(delta_reward, 4)

        # ── 3. Action Shaping ─────────────────────────────────────────────────
        action_bonus = self.ACTION_BONUS.get(action_type, 0.0) * self.W_ACTION
        components["action_bonus"] = round(action_bonus, 4)

        # ── 4. Milestone Bonuses ──────────────────────────────────────────────
        milestone_bonus = 0.0
        for metric, threshold, bonus, label in self.MILESTONES:
            key = f"{metric}_{threshold}"
            val = getattr(obs, metric, 0.0)

            # Pollution milestone: reward when BELOW threshold
            if metric == "pollution":
                crossed = val < threshold
            else:
                crossed = val >= threshold

            if crossed and key not in self.milestones_hit:
                self.milestones_hit.add(key)
                milestone_bonus += bonus
                events.append(f"🏅 Milestone: {label} (+{bonus:.2f})")

        components["milestone"] = round(milestone_bonus, 4)

        # ── 5. Penalty Signals ────────────────────────────────────────────────
        penalty = 0.0

        # Penalty for high temperature (overheating servers)
        if obs.temperature > 0.6:
            penalty += 0.10
            events.append("🔥 High temperature penalty")

        # Penalty for critical pollution
        if obs.pollution > 0.7:
            penalty += 0.08
            events.append("☠️ Critical pollution penalty")

        # Penalty for low uptime
        if obs.uptime < 0.5:
            penalty += 0.08
            events.append("💀 Low uptime penalty")

        # Penalty for repeated develop actions (destructive)
        if action_type == "develop":
            penalty += 0.05
            events.append("⚠️ Develop action penalty")

        components["penalty"] = round(-penalty, 4)

        # ── 6. Time Efficiency Bonus ──────────────────────────────────────────
        time_bonus = 0.0
        if obs.eco_score >= 0.85:
            # Reward finishing early
            time_remaining = (max_steps - step) / max_steps
            time_bonus = 0.10 * time_remaining
            events.append(f"⏱️ Time efficiency bonus: +{time_bonus:.3f}")

        components["time_bonus"] = round(time_bonus, 4)

        # ── Final Total ───────────────────────────────────────────────────────
        total = (
            eco_reward       +
            pollution_reward +
            temp_reward      +
            renewable_reward +
            uptime_reward    +
            delta_reward     +
            action_bonus     +
            milestone_bonus  +
            time_bonus       -
            penalty
        )

        total = round(min(1.0, max(0.0, total)), 4)
        components["total"] = total

        return {
            "reward":     total,
            "components": components,
            "events":     events,
        }


# ── Task Grader ───────────────────────────────────────────────────────────────
class TaskGrader:
    """Grades all 3 tasks with deterministic criteria."""

    @staticmethod
    def grade_all(obs, step: int, history: List[dict]) -> Dict[str, float]:
        eco        = getattr(obs, "eco_score",       0.0)
        pollution  = getattr(obs, "pollution",        1.0)
        renewable  = getattr(obs, "renewable_ratio",  0.0)
        temp       = getattr(obs, "temperature",      1.0)
        uptime     = getattr(obs, "uptime",           0.0)

        # ── Task Easy: Basic Monitoring & Awareness ───────────────────────────
        # Score based on: steps taken + eco above baseline
        monitor_steps = sum(
            1 for h in history if h.get("action") == "monitor"
        )
        task_easy = round(min(1.0,
            monitor_steps / 5.0 * 0.5 +
            min(eco / 0.4, 1.0) * 0.5
        ), 4)

        # ── Task Medium: Pollution & Temperature Control ───────────────────────
        pol_score  = 1.0 if pollution < 0.3 else max(0.0, (0.7 - pollution) / 0.7)
        temp_score = 1.0 if temp < 0.2 else max(0.0, (0.6 - temp) / 0.6)
        eco_score  = min(eco / 0.5, 1.0)
        task_medium = round(
            pol_score  * 0.35 +
            temp_score * 0.30 +
            eco_score  * 0.35,
        4)

        # ── Task Hard: Full Ecosystem Recovery ────────────────────────────────
        if eco >= 0.85:
            efficiency  = max(0.0, (20 - step) / 20.0)
            ren_bonus   = min(renewable / 0.5, 1.0) * 0.1
            task_hard   = round(min(1.0, 0.75 + 0.15 * efficiency + ren_bonus), 4)
        else:
            # Partial credit based on how close to target
            eco_progress = eco / 0.85
            ren_progress = min(renewable / 0.3, 1.0)
            task_hard    = round(
                eco_progress * 0.70 +
                ren_progress * 0.30,
            4)

        return {
            "task_easy":   task_easy,
            "task_medium": task_medium,
            "task_hard":   task_hard,
        }


# ── Singleton Reward Engine & Grader ──────────────────────────────────────────
reward_engine = RewardEngine()
task_grader   = TaskGrader()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "EcoServer Environment API v2.0",
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
    global env, current_obs, prev_obs, step_count, episode_history

    try:
        env             = EcoServerEnv(width=15, height=15, max_steps=50)
        current_obs     = env.reset()
        prev_obs        = None
        step_count      = 0
        episode_history = []
        reward_engine.reset()

        return {
            "status":      "ok",
            "observation": safe_obs(current_obs),
            "message":     "Environment reset successfully",
        }
    except Exception as e:
        return {"status": "ok", "observation": {}, "message": str(e)}


@app.post("/step")
async def step_env(action: ActionRequest):
    """Execute one step. Required by OpenEnv spec."""
    global env, current_obs, prev_obs, step_count, episode_history

    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")

    try:
        prev_obs = current_obs

        action_obj = EcoServerAction(
            action_type=action.action_type,
            x=action.x,
            y=action.y,
        )
        result     = env.step(action_obj)
        current_obs = result
        step_count += 1

        # Compute rich reward
        reward_info = reward_engine.compute(
            obs=result,
            prev_obs=prev_obs,
            action_type=action.action_type,
            step=step_count,
            max_steps=50,
        )

        # Grade all tasks
        episode_history.append({"action": action.action_type, "step": step_count})
        grades = task_grader.grade_all(result, step_count, episode_history)

        obs_dict = safe_obs(result)

        return {
            "observation": obs_dict,
            "reward":      reward_info["reward"],
            "done":        bool(result.done),
            "info": {
                "step":             step_count,
                "reward_breakdown": reward_info["components"],
                "reward_events":    reward_info["events"],
                "task_grades":      grades,
                "eco_score":        result.eco_score,
                "pollution":        result.pollution,
                "temperature":      result.temperature,
                "renewable_ratio":  result.renewable_ratio,
                "uptime":           result.uptime,
                "last_event":       result.last_event,
            },
            "message": "Episode done!" if result.done else "Step executed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state():
    """Return current state. Required by OpenEnv spec."""
    grades = task_grader.grade_all(
        current_obs, step_count, episode_history
    ) if current_obs else {}

    return {
        "observation":   safe_obs(current_obs),
        "step":          step_count,
        "env_initialized": env is not None,
        "task_grades":   grades,
        "message":       "Current state retrieved",
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
