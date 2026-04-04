from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Simplified import - Since all files are in the main directory now, 
# we don't need the sys.path hack or the "server." prefix!
from server.eco_server_env_environment import EcoServerEnv, EcoServerAction

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
            "state": "GET /state - Get current state",
            "health": "GET /health - Health check"
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

# --- ADDED FOR SCALER OPENENV HACKATHON VALIDATOR ---
def main():
    import uvicorn
    # This ensures the autograder starts your app on the correct port
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
