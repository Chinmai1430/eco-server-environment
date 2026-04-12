# inference.py
"""
EcoServer Baseline Inference Script.
Uses Scaler's LiteLLM proxy (API_BASE_URL + API_KEY) to make
LLM-driven decisions, prints [START]/[STEP]/[END] to stdout.
"""
import sys
import os
import time
import json
import requests

# ── MUST: flush=True on ALL prints ────────────────────────────────────────────
def log(msg: str):
    print(msg, flush=True)

# ── Environment Variables injected by Scaler ──────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY",      "dummy-key")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")

log(f"🔑 API_BASE_URL: {API_BASE_URL}")
log(f"🌍 ENV_BASE_URL: {ENV_BASE_URL}")
log(f"🤖 MODEL:        {MODEL_NAME}")

# ── Install openai if missing ─────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    log("⚠️ openai not installed — installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "openai", "-q"])
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        log("❌ Could not install openai")

# ── OpenAI Client via Scaler's proxy ─────────────────────────────────────────
if OPENAI_AVAILABLE:
    client = OpenAI(
        base_url=API_BASE_URL,   # ← Scaler's LiteLLM proxy
        api_key=API_KEY,         # ← Scaler's injected API key
    )

# ── Valid actions ─────────────────────────────────────────────────────────────
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

MAX_RETRIES = 15
RETRY_DELAY = 3


# ── Wait for server ───────────────────────────────────────────────────────────
def wait_for_server() -> bool:
    log(f"⏳ Waiting for server at {ENV_BASE_URL} ...")
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                log("✅ Server is ready!")
                return True
        except Exception:
            pass
        log(f"   Retry {i+1}/{MAX_RETRIES} ...")
        time.sleep(RETRY_DELAY)
    log("⚠️ Proceeding anyway...")
    return False


# ── API Helpers ───────────────────────────────────────────────────────────────
def reset_env() -> dict:
    try:
        r = requests.post(f"{ENV_BASE_URL}/reset", timeout=15)
        return r.json()
    except Exception as e:
        log(f"⚠️ Reset error: {e}")
        return {}


def step_env(action_type: str, x: int = 7, y: int = 7) -> dict:
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action_type": action_type, "x": x, "y": y},
            timeout=15,
        )
        return r.json()
    except Exception as e:
        log(f"⚠️ Step error: {e}")
        return {"reward": 0.0, "done": False, "info": {}, "observation": {}}


def get_state() -> dict:
    try:
        r = requests.get(f"{ENV_BASE_URL}/state", timeout=15)
        return r.json()
    except Exception as e:
        log(f"⚠️ State error: {e}")
        return {}


# ── LLM Agent (uses Scaler's proxy) ──────────────────────────────────────────
def llm_choose_action(obs: dict, task_name: str, step_num: int) -> tuple:
    """
    Use LLM via Scaler's LiteLLM proxy to choose the best action.
    Falls back to rule-based if LLM fails.
    """
    if not OPENAI_AVAILABLE:
        return rule_based_action(obs)

    eco        = obs.get("eco_score",       0.0)
    pollution  = obs.get("pollution",       0.0)
    temp       = obs.get("temperature",     0.0)
    renewable  = obs.get("renewable_ratio", 0.0)
    green      = obs.get("green_cover",     0.0)
    uptime     = obs.get("uptime",          1.0)

    prompt = f"""You are an AI agent managing an ecological server data center.
Current environment state at step {step_num} for task '{task_name}':
- eco_score: {eco:.3f} (higher is better, target >= 0.85)
- pollution: {pollution:.3f} (lower is better, target < 0.3)
- temperature: {temp:.3f} (lower is better, target < 0.2)
- renewable_ratio: {renewable:.3f} (higher is better, target >= 0.5)
- green_cover: {green:.3f} (higher is better)
- uptime: {uptime:.3f} (higher is better)

Available actions: {', '.join(VALID_ACTIONS)}

Grid is 15x15. Choose the single best action and coordinates.
Reply with ONLY a JSON object like this (no markdown, no explanation):
{{"action": "cool_server", "x": 7, "y": 7, "reason": "temperature is high"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ecological data center manager. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100,
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        # Clean markdown if present
        content = content.replace("```json", "").replace("```", "").strip()
        data    = json.loads(content)

        action = data.get("action", "monitor")
        x      = int(data.get("x", 7))
        y      = int(data.get("y", 7))

        # Validate
        if action not in VALID_ACTIONS:
            action = "monitor"
        x = max(0, min(x, 14))
        y = max(0, min(y, 14))

        log(f"   🤖 LLM chose: {action}({x},{y}) — {data.get('reason','')}")
        return action, x, y

    except Exception as e:
        log(f"   ⚠️ LLM error: {e} — using rule-based fallback")
        return rule_based_action(obs)


# ── Rule-Based Fallback ───────────────────────────────────────────────────────
def rule_based_action(obs: dict) -> tuple:
    temp      = obs.get("temperature",     0.0)
    pollution = obs.get("pollution",       0.0)
    renewable = obs.get("renewable_ratio", 0.0)
    green     = obs.get("green_cover",     0.0)
    eco       = obs.get("eco_score",       0.0)

    if temp      > 0.3: return "cool_server",        7, 7
    if pollution > 0.2: return "remove_pollution",   5, 5
    if renewable < 0.3: return "install_solar",      3, 3
    if green     < 0.3: return "plant_tree",         10, 10
    if eco       < 0.6: return "upgrade_efficiency", 7, 3
    return "monitor", 7, 7


# ── Task Runner ───────────────────────────────────────────────────────────────
def run_task(task_name: str, max_steps: int) -> dict:
    # ── REQUIRED: [START] block ───────────────────────────────────────────────
    log(f"[START] task={task_name}")

    reset_resp   = reset_env()
    obs          = reset_resp.get("observation", {})
    total_reward = 0.0
    final_score  = 0.0
    steps_taken  = 0

    for i in range(max_steps):
        step_num = i + 1

        # LLM decides action via Scaler's proxy
        action_type, x, y = llm_choose_action(obs, task_name, step_num)

        result      = step_env(action_type, x, y)
        reward      = float(result.get("reward",  0.0))
        done        = bool(result.get("done",     False))
        info        = result.get("info",          {})
        obs         = result.get("observation",   {})
        grades      = info.get("task_grades",     {})
        eco_score   = info.get("eco_score",   obs.get("eco_score",   0.0))
        pollution   = info.get("pollution",   obs.get("pollution",   0.0))
        temperature = info.get("temperature", obs.get("temperature", 0.0))

        total_reward += reward
        steps_taken   = step_num
        final_score   = grades.get(task_name, final_score)

        # ── REQUIRED: [STEP] block ────────────────────────────────────────────
        log(
            f"[STEP] step={steps_taken} "
            f"action={action_type} "
            f"reward={round(reward, 4)} "
            f"score={round(final_score, 4)} "
            f"eco={round(float(eco_score), 4)} "
            f"pollution={round(float(pollution), 4)} "
            f"temperature={round(float(temperature), 4)} "
            f"done={done}"
        )

        if done:
            break

    # ── REQUIRED: [END] block ─────────────────────────────────────────────────
    log(
        f"[END] task={task_name} "
        f"score={round(final_score, 4)} "
        f"steps={steps_taken} "
        f"total_reward={round(total_reward, 4)}"
    )

    return {
        "task":         task_name,
        "score":        round(final_score,   4),
        "total_reward": round(total_reward,  4),
        "steps":        steps_taken,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("🌍 EcoServer LLM Baseline Inference")
    log("=" * 60)

    wait_for_server()

    all_scores = {}

    # Task Easy
    try:
        result = run_task("task_easy", max_steps=10)
        all_scores["task_easy"] = result["score"]
    except Exception as e:
        log(f"[END] task=task_easy score=0.0 steps=0 total_reward=0.0")
        all_scores["task_easy"] = 0.0
        log(f"⚠️ task_easy failed: {e}")

    # Task Medium
    try:
        result = run_task("task_medium", max_steps=10)
        all_scores["task_medium"] = result["score"]
    except Exception as e:
        log(f"[END] task=task_medium score=0.0 steps=0 total_reward=0.0")
        all_scores["task_medium"] = 0.0
        log(f"⚠️ task_medium failed: {e}")

    # Task Hard
    try:
        result = run_task("task_hard", max_steps=20)
        all_scores["task_hard"] = result["score"]
    except Exception as e:
        log(f"[END] task=task_hard score=0.0 steps=0 total_reward=0.0")
        all_scores["task_hard"] = 0.0
        log(f"⚠️ task_hard failed: {e}")

    # Final summary
    log("=" * 60)
    log("🏆 FINAL BASELINE SCORES")
    log("=" * 60)
    for task, score in all_scores.items():
        bar = "█" * int(score * 20)
        log(f"  {task:15s}: {score:.4f}  |{bar:<20}|")
    log(json.dumps(all_scores, indent=2))
    log("✅ Inference completed successfully!")

    sys.exit(0)


if __name__ == "__main__":
    main()
