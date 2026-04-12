# test_local.py
# Run: python test_local.py
# Make sure app.py is running first: python app.py

import requests
import json

BASE_URL = "http://localhost:7860"

def sep(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)

def check(label, condition, value=""):
    status = "✅" if condition else "❌"
    print(f"  {status} {label} {value}")
    return condition

def hit(method, path, body=None):
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        else:
            r = requests.post(url, json=body or {}, timeout=30)
        return r
    except Exception as e:
        print(f"  ❌ Request failed: {e}")
        return None

# ── HEALTH ────────────────────────────────────────────────────────
sep("Health Check")
r = hit("GET", "/health")
if r:
    check("HTTP 200", r.status_code == 200, f"(got {r.status_code})")
    check("JSON parseable", True)
    print(f"  → {r.json()}")

# ── RESET — all 4 tasks ───────────────────────────────────────────
for task_id in ["easy", "medium", "hard", "loss"]:
    sep(f"POST /reset  task={task_id}")
    r = hit("POST", "/reset", {"task_id": task_id})
    if not r:
        continue
    check("HTTP 200", r.status_code == 200, f"(got {r.status_code})")
    if r.status_code != 200:
        print(f"  → {r.text[:300]}")
        continue
    obs = r.json()
    check("has 'step'",             "step"             in obs)
    check("has 'task_id'",          "task_id"          in obs)
    check("has 'pipeline_state'",   "pipeline_state"   in obs)
    check("has 'data_summary'",     "data_summary"     in obs)
    check("has 'training_metrics'", "training_metrics" in obs)
    check("has 'available_actions'","available_actions" in obs)
    check("has 'done'",             "done"             in obs)
    check("has 'confidence_score'", "confidence_score" in obs)
    check("done=False",             obs.get("done") == False)
    print(f"  → confidence_score={obs.get('confidence_score')}")
    print(f"  → available_actions={obs.get('available_actions')}")

# ── STEP — test every action on loss task ─────────────────────────
sep("POST /step — loss task full walkthrough")

hit("POST", "/reset", {"task_id": "loss"})

actions = [
    ("inspect_data",      None),
    ("inspect_metrics",   None),
    ("inspect_config",    None),
    ("inspect_model",     None),
    ("fix_loss_function", None),
    ("retrain",           None),
    ("submit_diagnosis",  None),
]

for action_type, parameter in actions:
    body = {"action_type": action_type, "reasoning": "test"}
    if parameter:
        body["parameter"] = parameter

    r = hit("POST", "/step", body)
    if not r:
        break

    ok = r.status_code == 200
    check(f"{action_type}", ok, f"(HTTP {r.status_code})")

    if not ok:
        print(f"  → ERROR: {r.text[:400]}")
        break

    data = r.json()
    obs  = data.get("observation", {})
    print(f"     reward={data.get('reward')}  done={data.get('done')}  "
          f"conf={obs.get('confidence_score')}  "
          f"val_acc={obs.get('training_metrics', {}).get('val_acc')}")

    if data.get("done"):
        print("  → Episode ended")
        break

# ── STATE ─────────────────────────────────────────────────────────
sep("GET /state")
r = hit("GET", "/state")
if r:
    check("HTTP 200", r.status_code == 200)
    s = r.json()
    check("has 'episode_id'",   "episode_id"   in s)
    check("has 'step'",         "step"         in s)
    check("has 'current_grade'","current_grade" in s)
    check("has 'bugs_fixed'",   "bugs_fixed"   in s)
    check("has 'done'",         "done"         in s)
    print(f"  → grade={s.get('current_grade')}  bugs_fixed={s.get('bugs_fixed')}")

# ── EDGE CASES ────────────────────────────────────────────────────
sep("Edge Cases")

# Empty body reset (validator does this)
r = hit("POST", "/reset", {})
check("Empty body {} returns 200", r and r.status_code == 200,
      f"(got {r.status_code if r else 'no response'})")

# Invalid task_id
r = hit("POST", "/reset", {"task_id": "invalid_task"})
check("Invalid task_id returns 400", r and r.status_code == 400,
      f"(got {r.status_code if r else 'no response'})")

print(f"\n{'='*50}")
print("  Done. Fix any ❌ before pushing to HF.")
print('='*50)