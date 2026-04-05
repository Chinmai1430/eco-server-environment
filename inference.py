# inference.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from server.eco_server_env_environment import EcoServerEnv, EcoServerAction
try:
    from visualization import visualize_grid, visualize_detailed_stats
except ImportError:
    def visualize_grid(grid): pass
    def visualize_detailed_stats(obs): pass

# ─── App & Global State ───────────────────────────────────────────────────────
app = FastAPI(title="EcoServer Inference API")

env: Optional[EcoServerEnv] = None
current_observation = None


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class ActionRequest(BaseModel):
    action_type: str          # "plant_tree" | "remove_pollution" | "monitor" | "develop"
    x: Optional[int] = None
    y: Optional[int] = None


class StepResponse(BaseModel):
    reward: float
    done: bool
    step: int
    eco_score: float
    message: str


# ─── Required OpenEnv Endpoints ───────────────────────────────────────────────

@app.post("/reset")
def reset():
    """
    OpenEnv required endpoint.
    Resets the environment and returns initial observation info.
    """
    global env, current_observation

    print("🔄 Resetting EcoServer Environment...")
    env = EcoServerEnv(width=15, height=15)
    current_observation = env.reset()

    visualize_grid(current_observation.grid)
    visualize_detailed_stats(current_observation)

    print("✅ Environment reset successfully!")
    return {
        "status": "ok",
        "message": "Environment reset successfully",
        "grid_width": env.width,
        "grid_height": env.height,
        "eco_score": float(getattr(current_observation, "eco_score", 0.0)),
    }


@app.post("/infer", response_model=StepResponse)
def infer(action_request: ActionRequest):
    """
    OpenEnv required endpoint.
    Accepts an action and returns the result of env.step().
    """
    global env, current_observation

    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )

    print(f"📍 Action received: {action_request.action_type} "
          f"at ({action_request.x}, {action_request.y})")

    action = EcoServerAction(
        action_type=action_request.action_type,
        x=action_request.x,
        y=action_request.y,
    )

    result = env.step(action)
    current_observation = result

    visualize_grid(result.grid)
    visualize_detailed_stats(result)

    return StepResponse(
        reward=float(result.reward),
        done=bool(result.done),
        step=int(getattr(result, "step", 0)),
        eco_score=float(getattr(result, "eco_score", 0.0)),
        message="Episode done!" if result.done else "Step executed successfully",
    )


@app.get("/health")
def health():
    """Optional health-check endpoint."""
    return {"status": "healthy", "env_initialized": env is not None}


# ─── Original CLI Demo (kept intact) ─────────────────────────────────────────

def main():
    print("🌍 Initializing EcoServer Environment...")

    demo_env = EcoServerEnv(width=15, height=15)
    observation = demo_env.reset()
    print("✅ Environment initialized!")

    visualize_grid(observation.grid)
    visualize_detailed_stats(observation)

    actions = [
        EcoServerAction(action_type="plant_tree", x=7, y=7),
        EcoServerAction(action_type="remove_pollution", x=5, y=5),
        EcoServerAction(action_type="monitor"),
        EcoServerAction(action_type="develop", x=3, y=3),
        EcoServerAction(action_type="plant_tree", x=10, y=10),
    ]

    for i, action in enumerate(actions):
        print(f"\n📍 Step {i + 1} - Action: {action.action_type}")
        result = demo_env.step(action)
        visualize_grid(result.grid)
        visualize_detailed_stats(result)
        print(f"💰 Reward: {result.reward:.1f}")
        if result.done:
            print("🏁 Episode completed!")
            break

    print("\n✅ Inference completed successfully!")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "demo"], default="server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)  # ← back to 7860
    args = parser.parse_args()

    if args.mode == "demo":
        main()
    else:
        print(f"🚀 Starting EcoServer API on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
