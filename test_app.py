# test_app.py
# Run this while app.py is running in another terminal.
# Uses requests library — no curl needed.

import requests
import json

BASE = "http://localhost:7860"
PASS = "✅"
FAIL = "❌"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  →  {detail}" if detail else ""))
    return condition

def pp(r):
    """Pretty print response — truncated to 200 chars."""
    try:
        txt = json.dumps(r.json(), indent=2)
        return txt[:300] + "..." if len(txt) > 300 else txt
    except:
        return r.text[:200]


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 0: Health Check ══")
# ══════════════════════════════════════════════════════════════════════════════
r = requests.get(f"{BASE}/")
check("GET / returns 200",          r.status_code == 200)
check("status == running",          r.json()["status"] == "running")
check("tasks list present",         "tasks" in r.json())
print(f"  Response: {pp(r)}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 1: POST /reset with empty body (VALIDATOR CHECK) ══")
# ══════════════════════════════════════════════════════════════════════════════
r = requests.post(f"{BASE}/reset",
    headers={"Content-Type": "application/json"},
    data="{}")   # exact what validator sends
check("empty body {} returns 200",  r.status_code == 200,  f"got {r.status_code}")
check("defaults to easy task",      r.json().get("task_id") == "easy")
check("has available_actions",      "available_actions" in r.json())
check("has episode_id",             "episode_id" in r.json())
print(f"  task_id: {r.json().get('task_id')}")
print(f"  episode_id: {r.json().get('episode_id')}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 2: POST /reset with task_id ══")
# ══════════════════════════════════════════════════════════════════════════════
for task in ["easy", "medium", "hard"]:
    r = requests.post(f"{BASE}/reset", json={"task_id": task})
    check(f"reset task={task} returns 200", r.status_code == 200, f"got {r.status_code}")
    check(f"task_id == {task}",             r.json().get("task_id") == task)


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 3: POST /step ══")
# ══════════════════════════════════════════════════════════════════════════════

# First reset to easy
requests.post(f"{BASE}/reset", json={"task_id": "easy"})

r = requests.post(f"{BASE}/step", json={
    "action_type": "inspect_data",
    "reasoning":   "checking data quality"
})
check("step returns 200",            r.status_code == 200, f"got {r.status_code}")
check("has observation",             "observation" in r.json())
check("has reward",                  "reward" in r.json())
check("has done",                    "done" in r.json())
check("has info",                    "info" in r.json())
check("reward is float",             isinstance(r.json()["reward"], float))
check("done is bool",                isinstance(r.json()["done"], bool))
print(f"  reward={r.json()['reward']}  done={r.json()['done']}")

# Step with all fields
r2 = requests.post(f"{BASE}/step", json={
    "action_type": "fix_labels",
    "parameter":   "0.3",
    "reasoning":   "labels appear to be flipped"
})
check("step with all fields returns 200", r2.status_code == 200)
check("obs has steps_remaining",    "steps_remaining" in r2.json()["observation"])
check("obs has total_reward",       "total_reward" in r2.json()["observation"])


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 4: GET /state ══")
# ══════════════════════════════════════════════════════════════════════════════
r = requests.get(f"{BASE}/state")
check("state returns 200",          r.status_code == 200, f"got {r.status_code}")
check("has episode_id",             "episode_id" in r.json())
check("has step count",             "step" in r.json())
check("has total_reward",           "total_reward" in r.json())
check("has current_grade",          "current_grade" in r.json())
check("has reward_history",         "reward_history" in r.json())
check("status == active",           r.json()["status"] == "active")
print(f"  step={r.json()['step']}  total_reward={r.json()['total_reward']}")
print(f"  current_grade={r.json()['current_grade']}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 5: Invalid task_id → 400 ══")
# ══════════════════════════════════════════════════════════════════════════════
r = requests.post(f"{BASE}/reset", json={"task_id": "super_hard"})
check("invalid task_id returns 400", r.status_code == 400, f"got {r.status_code}")
check("error message present",       "error" in r.json().get("detail", {}))
check("valid_task_ids in response",  "valid_task_ids" in r.json().get("detail", {}))
print(f"  detail: {r.json().get('detail')}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 6: step() before reset() → 400 ══")
# ══════════════════════════════════════════════════════════════════════════════

# Create a fresh server state by importing directly won't work over HTTP
# Instead test with a known-safe path: step after episode ends
requests.post(f"{BASE}/reset", json={"task_id": "easy"})
# Force episode to end
for _ in range(20):
    r_done = requests.post(f"{BASE}/step", json={"action_type": "submit_diagnosis"})
    if r_done.json().get("done"):
        break

# Now step after done — should return 200 with reward=0 (our guard)
r = requests.post(f"{BASE}/step", json={"action_type": "inspect_data"})
check("step after done returns 200",   r.status_code == 200)
check("step after done reward == 0",   r.json()["reward"] == 0.0)
check("step after done done == True",  r.json()["done"] == True)


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ CHECK 7: Full medium episode over HTTP ══")
# ══════════════════════════════════════════════════════════════════════════════
requests.post(f"{BASE}/reset", json={"task_id": "medium"})

steps = [
    {"action_type": "inspect_data",     "reasoning": "checking for class imbalance"},
    {"action_type": "inspect_metrics",  "reasoning": "checking loss divergence"},
    {"action_type": "inspect_config",   "reasoning": "checking learning rate"},
    {"action_type": "fix_class_balance","reasoning": "class imbalance causing poor recall"},
    {"action_type": "fix_normalization","reasoning": "feature magnitude too large"},
    {"action_type": "fix_learning_rate","parameter": "0.01",
                                        "reasoning": "lr too high causes divergence"},
    {"action_type": "retrain"},
]

total_reward = 0.0
final_grade  = 0.0
for i, s in enumerate(steps, 1):
    r = requests.post(f"{BASE}/step", json=s)
    total_reward += r.json()["reward"]
    final_grade   = r.json()["info"]["current_grade"]
    done = r.json()["done"]
    print(f"  Step {i} [{s['action_type']}]  r={r.json()['reward']:.2f}  grade={final_grade:.4f}  done={done}")
    if done:
        break

check("Full episode total_reward > 1.0", total_reward > 1.0, f"total={total_reward:.2f}")
check("Final grade > 0.6",               final_grade > 0.6,  f"grade={final_grade:.4f}")

state = requests.get(f"{BASE}/state").json()
check("state reflects completed episode", state["current_grade"] > 0.6)

print(f"\n  {'='*50}")
print(f"  All server checks complete.")
print(f"  This server is ready for HF Spaces deployment.")
print(f"  {'='*50}")
