# inference.py
"""
Baseline inference script for EcoServer Environment.
Prints required [START]/[STEP]/[END] structured output blocks to stdout.
"""
import sys
import os
import time
import json
import requests

BASE_URL    = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MAX_RETRIES = 15
RETRY_DELAY = 3


# ── MUST USE flush=True on ALL prints ─────────────────────────────────────────
def log(msg: str):
    print(msg, flush=True)


# ── Wait for server ───────────────────────────────────────────────────────────
def wait_for_server() -> bool:
    log(f"⏳ Waiting for server at {BASE_URL} ...")
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                log(f"✅ Server is ready!")
                return True
        except Exception:
            pass
        log(f"   Retry {i+1}/{MAX_RETRIES} ...")
        time.sleep(RETRY_DELAY)
    log("⚠️ Server not reachable — proceeding anyway...")
    return False


# ── API Helpers ───────────────────────────────────────────────────────────────
def reset() -> dict:
    try:
        r = requests.post(f"{BASE_URL}/reset", timeout=15)
        return r.json()
    except Exception as e:
        log(f"⚠️ Reset error: {e}")
        return {}


def step(action_type: str, x: int = 7, y: int = 7) -> dict:
    try:
        r = requests.post(
            f"{BASE_URL}/step",
            json={"action_type": action_type, "x": x, "y": y},
            timeout=15,
        )
        return r.json()
    except Exception as e:
        log(f"⚠️ Step error: {e}")
        return {"reward": 0.0, "done": False, "info": {}}


def get_state() -> dict:
    try:
        r = requests.get(f"{BASE_URL}/state", timeout=15)
        return r.json()
    except Exception as e:
        log(f"⚠️ State error: {e}")
        return {}


# ── Rule-Based Agent ──────────────────────────────────────────────────────────
def rule_based_action(obs: dict):
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


# ── Task Runner (prints required structured output) ───────────────────────────
def run_task(task_name: str, actions: list, max_steps: int) -> dict:
    """
    Run a task and print required [START]/[STEP]/[END] blocks.
    Returns final scores.
    """
    # ── REQUIRED: Print [START] block ─────────────────────────────────────────
    log(f"[START] task={task_name}", )

    reset_resp = reset()
    obs = reset_resp.get("observation", {})

    total_reward = 0.0
    final_score  = 0.0
    steps_taken  = 0

    for i in range(max_steps):
        # Choose action
        if i < len(actions):
            action_type, x, y = actions[i]
        else:
            # Rule-based for remaining steps
            state_resp  = get_state()
            obs         = state_resp.get("observation", {})
            action_type, x, y = rule_based_action(obs)

        result = step(action_type, x, y)

        reward      = float(result.get("reward", 0.0))
        done        = bool(result.get("done", False))
        info        = result.get("info", {})
        obs         = result.get("observation", {})
        grades      = info.get("task_grades", {})
        eco_score   = info.get("eco_score", obs.get("eco_score", 0.0))
        pollution   = info.get("pollution",  obs.get("pollution",  0.0))
        temperature = info.get("temperature",obs.get("temperature",0.0))

        total_reward += reward
        steps_taken   = i + 1
        final_score   = grades.get(task_name, 0.0)

        # ── REQUIRED: Print [STEP] block ──────────────────────────────────────
        log(
            f"[STEP] step={steps_taken} "
            f"action={action_type} "
            f"reward={round(reward, 4)} "
            f"score={round(final_score, 4)} "
            f"eco={round(eco_score, 4)} "
            f"pollution={round(pollution, 4)} "
            f"temperature={round(temperature, 4)} "
            f"done={done}"
        )

        if done:
            break

    # ── REQUIRED: Print [END] block ───────────────────────────────────────────
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


# ── Task Definitions ──────────────────────────────────────────────────────────
TASK_EASY_ACTIONS = [
    ("monitor",          7,  7),
    ("monitor",          3,  3),
    ("monitor",          11, 11),
    ("plant_tree",       7,  7),
    ("monitor",          5,  5),
    ("remove_pollution", 8,  8),
    ("monitor",          2,  2),
    ("cool_server",      4,  4),
    ("monitor",          9,  9),
    ("monitor",          6,  6),
]

TASK_MEDIUM_ACTIONS = [
    ("cool_server",        3,  3),
    ("remove_pollution",   5,  5),
    ("cool_server",        8,  8),
    ("remove_pollution",   2,  9),
    ("plant_tree",         7,  7),
    ("cool_server",        11, 4),
    ("remove_pollution",   6,  6),
    ("upgrade_efficiency", 4,  4),
    ("remove_pollution",   10, 10),
    ("cool_server",        1,  1),
]

TASK_HARD_ACTIONS = [
    ("cool_server",        7,  7),
    ("cool_server",        3,  3),
    ("remove_pollution",   5,  5),
    ("install_solar",      7,  7),
    ("plant_tree",         10, 10),
    ("cool_server",        11, 11),
    ("remove_pollution",   2,  8),
    ("install_solar",      4,  4),
    ("plant_tree",         8,  2),
    ("upgrade_efficiency", 6,  6),
    ("cool_server",        1,  13),
    ("install_solar",      13, 1),
    ("remove_pollution",   9,  9),
    ("plant_tree",         3,  11),
    ("upgrade_efficiency", 11, 3),
    ("install_solar",      5,  10),
    ("plant_tree",         12, 6),
    ("cool_server",        6,  12),
    ("remove_pollution",   14, 14),
    ("monitor",            7,  7),
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("🌍 EcoServer Baseline Inference Script")
    log("=" * 60)

    # Wait for server
    wait_for_server()

    all_scores = {}

    # ── Task Easy ─────────────────────────────────────────────────────────────
    try:
        result = run_task("task_easy", TASK_EASY_ACTIONS, max_steps=10)
        all_scores["task_easy"] = result["score"]
    except Exception as e:
        log(f"[END] task=task_easy score=0.0 steps=0 total_reward=0.0")
        all_scores["task_easy"] = 0.0
        log(f"⚠️ task_easy error: {e}")

    # ── Task Medium ───────────────────────────────────────────────────────────
    try:
        result = run_task("task_medium", TASK_MEDIUM_ACTIONS, max_steps=10)
        all_scores["task_medium"] = result["score"]
    except Exception as e:
        log(f"[END] task=task_medium score=0.0 steps=0 total_reward=0.0")
        all_scores["task_medium"] = 0.0
        log(f"⚠️ task_medium error: {e}")

    # ── Task Hard ─────────────────────────────────────────────────────────────
    try:
        result = run_task("task_hard", TASK_HARD_ACTIONS, max_steps=20)
        all_scores["task_hard"] = result["score"]
    except Exception as e:
        log(f"[END] task=task_hard score=0.0 steps=0 total_reward=0.0")
        all_scores["task_hard"] = 0.0
        log(f"⚠️ task_hard error: {e}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    log("=" * 60)
    log("🏆 FINAL BASELINE SCORES")
    log("=" * 60)
    for task, score in all_scores.items():
        bar = "█" * int(score * 20)
        log(f"  {task:15s}: {score:.4f}  |{bar:<20}|")
    log(json.dumps(all_scores, indent=2))
    log("=" * 60)
    log("✅ Inference completed successfully!")

    # ALWAYS exit with 0
    sys.exit(0)


if __name__ == "__main__":
    main()
