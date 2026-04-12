# inference.py
"""
EcoServer Baseline Inference Script.
- Uses Scaler's LiteLLM proxy (API_BASE_URL + API_KEY)
- Prints [START]/[STEP]/[END] structured blocks to stdout
- Never starts a server (pure HTTP client)
- Always exits with code 0
"""
import sys, os, time, json, requests

def log(msg: str):
    print(msg, flush=True)

# ── Scaler-injected environment variables ─────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY",      "dummy-key")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")

log(f"🔑 API_BASE_URL : {API_BASE_URL}")
log(f"🌍 ENV_BASE_URL : {ENV_BASE_URL}")
log(f"🤖 MODEL        : {MODEL_NAME}")

# ── Install openai if missing ─────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    import subprocess
    subprocess.run([sys.executable,"-m","pip","install","openai","-q"], check=False)
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
    except:
        OPENAI_AVAILABLE = False
        log("❌ openai unavailable — using rule-based fallback")

# ── OpenAI client via Scaler's LiteLLM proxy ─────────────────────────────────
if OPENAI_AVAILABLE:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

VALID_ACTIONS = [
    "plant_tree","remove_pollution","monitor","develop",
    "install_solar","cool_server","upgrade_efficiency","decommission",
]

# ── Wait for server ───────────────────────────────────────────────────────────
def wait_for_server(retries=15, delay=3) -> bool:
    log(f"⏳ Waiting for server at {ENV_BASE_URL} ...")
    for i in range(retries):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                log("✅ Server ready!")
                return True
        except:
            pass
        log(f"   Retry {i+1}/{retries}...")
        time.sleep(delay)
    log("⚠️ Proceeding anyway...")
    return False

# ── API helpers ───────────────────────────────────────────────────────────────
def reset_env() -> dict:
    try:
        return requests.post(f"{ENV_BASE_URL}/reset", timeout=15).json()
    except Exception as e:
        log(f"⚠️ reset error: {e}")
        return {}

def step_env(action_type: str, x: int = 7, y: int = 7) -> dict:
    try:
        return requests.post(f"{ENV_BASE_URL}/step",
            json={"action_type":action_type,"x":x,"y":y}, timeout=15).json()
    except Exception as e:
        log(f"⚠️ step error: {e}")
        return {"reward":0.0,"done":False,"info":{},"observation":{}}

def get_state() -> dict:
    try:
        return requests.get(f"{ENV_BASE_URL}/state", timeout=15).json()
    except:
        return {}

# ── Rule-based fallback ───────────────────────────────────────────────────────
def rule_based(obs: dict):
    t = obs.get("temperature",     0.0)
    p = obs.get("pollution",       0.0)
    r = obs.get("renewable_ratio", 0.0)
    g = obs.get("green_cover",     0.0)
    e = obs.get("eco_score",       0.0)
    if t > 0.3: return "cool_server",        7, 7
    if p > 0.2: return "remove_pollution",   5, 5
    if r < 0.3: return "install_solar",      3, 3
    if g < 0.3: return "plant_tree",        10,10
    if e < 0.6: return "upgrade_efficiency", 7, 3
    return "monitor", 7, 7

# ── LLM agent via Scaler's proxy ─────────────────────────────────────────────
def llm_action(obs: dict, task: str, step_num: int):
    if not OPENAI_AVAILABLE:
        return rule_based(obs)
    try:
        prompt = f"""You are an AI agent managing an ecological data center grid (15x15).

Task: {task} | Step: {step_num}
State:
  eco_score:       {obs.get('eco_score',0):.3f}  (target >= 0.85)
  pollution:       {obs.get('pollution',0):.3f}   (target < 0.30)
  temperature:     {obs.get('temperature',0):.3f} (target < 0.20)
  renewable_ratio: {obs.get('renewable_ratio',0):.3f} (target >= 0.50)
  green_cover:     {obs.get('green_cover',0):.3f}
  uptime:          {obs.get('uptime',1):.3f}

Actions: {', '.join(VALID_ACTIONS)}
Grid coordinates: x,y in range 0-14.

Respond ONLY with valid JSON (no markdown):
{{"action": "cool_server", "x": 7, "y": 7, "reason": "temperature is high"}}"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role":"system","content":"You are an expert ecological data center manager. Respond with valid JSON only."},
                {"role":"user","content":prompt}
            ],
            max_tokens=120,
            temperature=0.1,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = raw.replace("```json","").replace("```","").strip()
        data = json.loads(raw)

        action = data.get("action","monitor")
        x      = max(0, min(int(data.get("x", 7)), 14))
        y      = max(0, min(int(data.get("y", 7)), 14))
        if action not in VALID_ACTIONS:
            action = "monitor"

        log(f"   🤖 LLM: {action}({x},{y}) — {data.get('reason','')}")
        return action, x, y

    except Exception as e:
        log(f"   ⚠️ LLM fallback: {e}")
        return rule_based(obs)

# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task_name: str, max_steps: int) -> dict:
    # ── REQUIRED ─────────────────────────────────────────────────────────────
    log(f"[START] task={task_name}")

    resp         = reset_env()
    obs          = resp.get("observation", {})
    total_reward = 0.0
    final_score  = 0.0
    steps_taken  = 0

    for i in range(max_steps):
        step_num    = i + 1
        action, x, y = llm_action(obs, task_name, step_num)
        result      = step_env(action, x, y)

        reward      = float(result.get("reward",  0.0))
        done        = bool(result.get("done",     False))
        info        = result.get("info",          {})
        obs         = result.get("observation",   obs)
        grades      = info.get("task_grades",     {})
        eco         = info.get("eco_score",   obs.get("eco_score",   0.0))
        pol         = info.get("pollution",   obs.get("pollution",   0.0))
        tmp         = info.get("temperature", obs.get("temperature", 0.0))

        total_reward += reward
        steps_taken   = step_num
        final_score   = grades.get(task_name, final_score)

        # ── REQUIRED ─────────────────────────────────────────────────────────
        log(f"[STEP] step={steps_taken} action={action} "
            f"reward={round(reward,4)} score={round(final_score,4)} "
            f"eco={round(float(eco),4)} pollution={round(float(pol),4)} "
            f"temperature={round(float(tmp),4)} done={done}")

        if done:
            break

    # ── REQUIRED ─────────────────────────────────────────────────────────────
    log(f"[END] task={task_name} score={round(final_score,4)} "
        f"steps={steps_taken} total_reward={round(total_reward,4)}")

    return {"task":task_name,"score":round(final_score,4),
            "total_reward":round(total_reward,4),"steps":steps_taken}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log("="*60)
    log("🌍 EcoServer LLM Baseline Inference")
    log("="*60)

    wait_for_server()

    scores = {}

    for task_name, max_steps in [
        ("task_easy",   10),
        ("task_medium", 10),
        ("task_hard",   20),
    ]:
        try:
            result = run_task(task_name, max_steps)
            scores[task_name] = result["score"]
        except Exception as e:
            log(f"[END] task={task_name} score=0.0 steps=0 total_reward=0.0")
            scores[task_name] = 0.0
            log(f"⚠️ {task_name} error: {e}")

    log("="*60)
    log("🏆 FINAL BASELINE SCORES")
    log("="*60)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        log(f"  {task:15s}: {score:.4f}  |{bar:<20}|")
    log(json.dumps(scores, indent=2))
    log("✅ Inference completed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
