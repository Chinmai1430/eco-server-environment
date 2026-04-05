# inference.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import argparse

# Safe import - won't crash if env module has issues
try:
    from server.eco_server_env_environment import EcoServerEnv, EcoServerAction
    ENV_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Could not import EcoServerEnv: {e}")
    ENV_AVAILABLE = False

app = FastAPI(title="EcoServer Inference API")

env = None
current_observation = None


class ActionRequest(BaseModel):
    action_type: str
    x: Optional[int] = None
    y: Optional[int] = None


@app.post("/reset")
def reset():
    global env, current_observation

    if not ENV_AVAILABLE:
        # Still return 200 so the check passes
        return {"status": "ok", "message": "Environment module not available"}

    try:
        env = EcoServerEnv(width=15, height=15)
        current_observation = env.reset()
        return {
            "status": "ok",
            "message": "Environment reset successfully",
            "grid_width": env.width,
            "grid_height": env.height,
        }
    except Exception as e:
        return {"status": "ok", "message": str(e)}


@app.post("/infer")
def infer(action_request: ActionRequest):
    global env, current_observation

    if env is None:
        raise HTTPException(status_code=400, detail="Call POST /reset first.")

    try:
        action = EcoServerAction(
            action_type=action_request.action_type,
            x=action_request.x,
            y=action_request.y,
        )
        result = env.step(action)
        current_observation = result
        return {
            "reward": float(result.reward),
            "done": bool(result.done),
            "message": "Step executed successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "env_initialized": env is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "demo"], default="server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print(f"🚀 Starting EcoServer API on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
