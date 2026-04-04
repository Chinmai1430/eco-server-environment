# server/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .eco_server_env_environment import EcoServerEnv, EcoServerAction
import json

app = FastAPI(title="EcoServer Environment API")

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment
env = EcoServerEnv()

@app.get("/")
async def root():
    return {
        "message": "EcoServer Environment API",
        "endpoints": {
            "reset": "POST /reset - Reset environment",
            "step": "POST /step - Take action", 
            "state": "GET /state - Get current state"
        }
    }

@app.post("/reset")
async def reset_env():
    """Reset the environment to initial state"""
    obs = env.reset()
    return {
        "observation": obs.__dict__,
        "message": "Environment reset successfully"
    }

@app.post("/step") 
async def step_env(action: dict):
    """Take an action in the environment"""
    try:
        # Convert dict to EcoServerAction
        action_obj = EcoServerAction(**action)
        result = env.step(action_obj)
        
        return {
            "observation": result.__dict__,
            "reward": result.reward,
            "done": result.done,
            "message": "Action executed successfully"
        }
    except Exception as e:
        return {"error": str(e), "message": "Failed to execute action"}

@app.get("/state")
async def get_state():
    """Get current environment state"""
    state = env.state
    return {
        "state": state.__dict__,
        "message": "Current state retrieved"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "EcoServer environment is running"}
