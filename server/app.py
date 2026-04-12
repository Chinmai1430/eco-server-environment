# server/app.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional, Any, Dict, List

from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

app = FastAPI(
    title="EcoServer Environment API",
    version="2.0.0",
    description="Real-world ecological data center management environment for OpenEnv.",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Global State ──────────────────────────────────────────────────────────────
env             = None
current_obs     = None
prev_obs        = None
step_count      = 0
episode_history: List[dict] = []
milestones_hit  = set()

VALID_ACTIONS = [
    "plant_tree","remove_pollution","monitor","develop",
    "install_solar","cool_server","upgrade_efficiency","decommission",
]

# ── Models ────────────────────────────────────────────────────────────────────
class ActionRequest(BaseModel):
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None

    @validator("action_type")
    def validate_action(cls, v):
        return v if v in VALID_ACTIONS else "monitor"

    @validator("x", "y", pre=True, always=True)
    def validate_coord(cls, v):
        if v is None: return None
        try: return max(0, min(int(v), 14))
        except: return None

# ── Exception Handler ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content={
        "status": "ok", "error": str(exc),
        "observation": safe_obs(current_obs),
        "reward": 0.0, "done": False,
        "info": {"step": step_count},
        "message": "Error handled gracefully",
    })

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_obs(obs) -> Dict[str, Any]:
    if obs is None:
        return {
            "eco_score":0.0,"pollution":0.5,"green_cover":0.0,
            "temperature":0.5,"energy_usage":0.5,"renewable_ratio":0.0,
            "server_load":0.5,"uptime":0.5,"step":step_count,
            "done":False,"reward":0.0,"last_event":"","win":False,"lose":False,
        }
    result = {}
    for k, v in obs.__dict__.items():
        try:
            result[k] = v.tolist() if hasattr(v,"tolist") else v
        except:
            result[k] = str(v)
    return result

def compute_reward(obs, prev, action_type: str, step: int) -> Dict[str,Any]:
    global milestones_hit
    events, components = [], {}

    eco    = float(getattr(obs,"eco_score",      0.0))
    pol    = float(getattr(obs,"pollution",      1.0))
    temp   = float(getattr(obs,"temperature",    1.0))
    ren    = float(getattr(obs,"renewable_ratio",0.0))
    uptime = float(getattr(obs,"uptime",         0.0))

    components["eco"]         = round(eco    * 0.25, 4)
    components["pollution"]   = round((1-pol)* 0.15, 4)
    components["temperature"] = round((1-temp)*0.10, 4)
    components["renewable"]   = round(ren    * 0.15, 4)
    components["uptime"]      = round(uptime * 0.10, 4)

    delta = 0.0
    if prev:
        de = eco  - float(getattr(prev,"eco_score",     0))
        dp = float(getattr(prev,"pollution",     0)) - pol
        dt = float(getattr(prev,"temperature",   0)) - temp
        dr = ren  - float(getattr(prev,"renewable_ratio",0))
        delta = (de*0.4+dp*0.3+dt*0.15+dr*0.15)*0.20*5.0
        if de > 0.02: events.append(f"📈 Eco +{de:.3f}")
        if dp > 0.02: events.append(f"🧹 Pollution -{dp:.3f}")
    components["delta"] = round(delta, 4)

    ab = {"plant_tree":0.15,"remove_pollution":0.20,"install_solar":0.18,
          "cool_server":0.15,"upgrade_efficiency":0.12,"decommission":0.05,
          "monitor":0.03,"develop":-0.15}.get(action_type,0.0)*0.05
    components["action_bonus"] = round(ab, 4)

    mb = 0.0
    for key, cond, bonus, label in [
        ("eco50",  eco>=0.50,  0.05, "🌱 Eco>50%"),
        ("eco70",  eco>=0.70,  0.08, "🌿 Eco>70%"),
        ("eco85",  eco>=0.85,  0.12, "🌳 Eco>85%"),
        ("eco95",  eco>=0.95,  0.20, "🏆 Eco>95%"),
        ("pol30",  pol<=0.30,  0.06, "✨ Pollution<30%"),
        ("pol10",  pol<=0.10,  0.10, "💎 Pollution<10%"),
        ("ren50",  ren>=0.50,  0.08, "☀️ 50% renewable"),
        ("ren80",  ren>=0.80,  0.12, "⚡ 80% renewable"),
    ]:
        if cond and key not in milestones_hit:
            milestones_hit.add(key)
            mb += bonus
            events.append(f"🏅 {label} +{bonus:.2f}")
    components["milestone"] = round(mb, 4)

    penalty = 0.0
    if temp   > 0.6: penalty+=0.10; events.append("🔥 Heat penalty")
    if pol    > 0.7: penalty+=0.08; events.append("☠️ Pollution penalty")
    if uptime < 0.5: penalty+=0.08; events.append("💀 Uptime penalty")
    if action_type=="develop": penalty+=0.05
    components["penalty"] = round(-penalty, 4)

    tb = 0.0
    if eco >= 0.85:
        tb = 0.10 * max(0.0,(50-step)/50.0)
        events.append(f"⏱️ Time bonus +{tb:.3f}")
    components["time_bonus"] = round(tb, 4)

    total = sum(components.values())
    total = round(min(1.0, max(0.0, total)), 4)
    components["total"] = total
    return {"reward": total, "components": components, "events": events}

def grade_tasks(obs, step: int, history: List[dict]) -> Dict[str,float]:
    if obs is None:
        return {"task_easy":0.0,"task_medium":0.0,"task_hard":0.0}
    eco  = float(getattr(obs,"eco_score",      0.0))
    pol  = float(getattr(obs,"pollution",      1.0))
    ren  = float(getattr(obs,"renewable_ratio",0.0))
    temp = float(getattr(obs,"temperature",    1.0))

    mon = sum(1 for h in history if h.get("action")=="monitor")
    task_easy = round(min(1.0, mon/5.0*0.5 + min(eco/0.4,1.0)*0.5), 4)

    ps = 1.0 if pol<0.3 else max(0.0,(0.7-pol)/0.7)
    ts = 1.0 if temp<0.2 else max(0.0,(0.6-temp)/0.6)
    es = min(eco/0.5, 1.0)
    task_medium = round(ps*0.35+ts*0.30+es*0.35, 4)

    if eco >= 0.85:
        eff = max(0.0,(20-step)/20.0)
        rb  = min(ren/0.5,1.0)*0.1
        task_hard = round(min(1.0, 0.75+0.15*eff+rb), 4)
    else:
        task_hard = round((eco/0.85)*0.70+min(ren/0.3,1.0)*0.30, 4)

    return {"task_easy":task_easy,"task_medium":task_medium,"task_hard":task_hard}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "EcoServer Environment",
        "version": "2.0.0",
        "description": (
            "Real-world AI environment for ecological data center management. "
            "Agent controls a 15x15 server grid making decisions about cooling, "
            "pollution removal, renewable energy deployment and tree planting "
            "to maximize ecological sustainability."
        ),
        "real_world_task": True,
        "domain": "Environmental sustainability / Data center management",
        "tasks": {
            "task_easy":   "Monitor grid 5 times (max 10 steps)",
            "task_medium": "Pollution<30%, temp<20%, eco>50% (max 10 steps)",
            "task_hard":   "Eco>=85%, renewable>=30% (max 20 steps)",
        },
        "action_space":      {"type":"discrete","n":8,"actions":VALID_ACTIONS},
        "observation_space": {"type":"dict","metrics":[
            "eco_score","pollution","green_cover","temperature",
            "energy_usage","renewable_ratio","server_load","uptime"
        ]},
        "reward_range": [0.0, 1.0],
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
    global env, current_obs, prev_obs, step_count, episode_history, milestones_hit
    try:
        env             = EcoServerEnv(width=15, height=15, max_steps=50)
        current_obs     = env.reset()
        prev_obs        = None
        step_count      = 0
        episode_history = []
        milestones_hit  = set()
        return {"status":"ok","observation":safe_obs(current_obs),"message":"Environment reset successfully"}
    except Exception as e:
        return {"status":"ok","observation":safe_obs(None),"message":str(e)}

@app.post("/step")
async def step_env(action: ActionRequest):
    global env, current_obs, prev_obs, step_count, episode_history
    if env is None:
        await reset_env()
    try:
        prev_obs   = current_obs
        result     = env.step(EcoServerAction(action_type=action.action_type,x=action.x,y=action.y))
        current_obs= result
        step_count += 1
        episode_history.append({"action":action.action_type,"step":step_count})

        ri     = compute_reward(result, prev_obs, action.action_type, step_count)
        grades = grade_tasks(result, step_count, episode_history)
        obs_d  = safe_obs(result)

        return {
            "observation": obs_d,
            "reward":      ri["reward"],
            "done":        bool(result.done),
            "info": {
                "step":             step_count,
                "eco_score":        result.eco_score,
                "pollution":        result.pollution,
                "temperature":      result.temperature,
                "renewable_ratio":  result.renewable_ratio,
                "green_cover":      result.green_cover,
                "uptime":           result.uptime,
                "win":              result.win,
                "lose":             result.lose,
                "last_event":       result.last_event,
                "reward_breakdown": ri["components"],
                "reward_events":    ri["events"],
                "task_grades":      grades,
            },
            "message": "Episode done!" if result.done else "Step executed",
        }
    except Exception as e:
        return {"observation":safe_obs(current_obs),"reward":0.0,"done":False,
                "info":{"step":step_count,"error":str(e)},"message":str(e)}

@app.get("/state")
async def get_state():
    try:
        return {
            "observation":     safe_obs(current_obs),
            "step":            step_count,
            "env_initialized": env is not None,
            "task_grades":     grade_tasks(current_obs, step_count, episode_history),
            "message":         "Current state retrieved",
        }
    except Exception as e:
        return {"observation":safe_obs(None),"step":step_count,"env_initialized":False,"task_grades":{},"message":str(e)}

@app.get("/health")
async def health():
    return {"status":"healthy","env_initialized":env is not None,"step":step_count,"version":"2.0.0"}

def main():
    """Main entry point. Required by openenv validate."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
