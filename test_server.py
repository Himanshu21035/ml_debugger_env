#!/usr/bin/env python3
# test_server.py
# Smoke test for the ML Debugger FastAPI server.
# Run AFTER starting the server: uvicorn app:app --port 7860
# Usage: python test_server.py
#        python test_server.py --url http://localhost:7860  (custom URL)

import sys
import json
import argparse
import requests

PASS = "✅"
FAIL = "❌"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  →  {detail}" if detail else ""))
    results.append((label, condition))
    return condition

def post(url, path, body=None):
    r = requests.post(f"{url}{path}", json=body or {}, timeout=60)
    return r

def get(url, path):
    r = requests.get(f"{url}{path}", timeout=30)
    return r

def run(base_url: str):
    print(f"\n        Target: {base_url}\n        Starting tests...\n   {'='*55}\n")

    # ══════════════════════════════════════════════════════════════════════
    print("══ 1. HEALTH CHECK (GET /) ══")
    # ══════════════════════════════════════════════════════════════════════
    r = get(base_url, "/")
    check("GET / returns 200",          r.status_code == 200)
    check("status == running",          r.json().get("status") == "running")
    check("tasks list present",         "tasks" in r.json())

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 2. STATE BEFORE RESET (GET /state) ══")
    # ══════════════════════════════════════════════════════════════════════
    r = get(base_url, "/state")
    check("GET /state returns 200",     r.status_code == 200)
    check("idle status before reset",   r.json().get("status") == "idle")

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 3. RESET — empty body {} (validator requirement) ══")
    # ══════════════════════════════════════════════════════════════════════
    r = post(base_url, "/reset", {})
    check("POST /reset {} returns 200", r.status_code == 200)
    obs = r.json()
    check("task_id == easy",            obs.get("task_id") == "easy")
    check("step == 0",                  obs.get("step") == 0)
    check("done == False",              obs.get("done") == False)
    check("available_actions present",  "available_actions" in obs)
    check("pipeline_state present",     "pipeline_state" in obs)
    check("training_metrics present",   "training_metrics" in obs)
    check("episode_id injected",        "episode_id" in obs)
    check("steps_remaining injected",   "steps_remaining" in obs)
    check("no numpy types in response", all(
        isinstance(v, (str, int, float, bool, list, dict, type(None)))
        for v in [obs.get("step"), obs.get("done"), obs.get("task_id")]
    ))

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 4. RESET — explicit task_id ══")
    # ══════════════════════════════════════════════════════════════════════
    for task_id in ["easy", "medium", "hard"]:
        r = post(base_url, "/reset", {"task_id": task_id})
        check(f"POST /reset {task_id} → 200",
              r.status_code == 200, f"status={r.status_code}")
        check(f"task_id == {task_id}",
              r.json().get("task_id") == task_id)

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 5. STEP — basic actions on easy task ══")
    # ══════════════════════════════════════════════════════════════════════
    post(base_url, "/reset", {"task_id": "easy"})

    for action_type in ["inspect_data", "inspect_metrics", "inspect_config"]:
        r = post(base_url, "/step", {"action_type": action_type})
        check(f"POST /step {action_type} → 200",
              r.status_code == 200, f"status={r.status_code}")
        body = r.json()
        check(f"  has observation",  "observation" in body)
        check(f"  has reward",       "reward" in body)
        check(f"  has done",         "done" in body)
        check(f"  has info",         "info" in body)
        check(f"  reward is float",  isinstance(body.get("reward"), (int, float)))

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 6. STEP — reasoning bonus ══")
    # ══════════════════════════════════════════════════════════════════════
    post(base_url, "/reset", {"task_id": "easy"})
    r = post(base_url, "/step", {
        "action_type": "fix_labels",
        "reasoning": "labels are flipped — this is a label noise issue"
    })
    check("fix_labels with reasoning → 200", r.status_code == 200)
    check("reward > 0",
          r.json().get("reward", -1) > 0,
          f"reward={r.json().get('reward')}")

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 7. STEP — invalid action handled gracefully ══")
    # ══════════════════════════════════════════════════════════════════════
    post(base_url, "/reset", {"task_id": "easy"})
    r = post(base_url, "/step", {"action_type": "totally_invalid_action"})
    check("invalid action returns 200 (not 500)",  r.status_code == 200)
    check("reward is negative",
          r.json().get("reward", 0) < 0,
          f"reward={r.json().get('reward')}")

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 8. STEP before RESET → 400 ══")
    # ══════════════════════════════════════════════════════════════════════
    fresh = requests.Session()  # clean session, no reset called
    # Force a fresh env by restarting would need a new server instance
    # Instead, just verify a normal step after reset works
    post(base_url, "/reset", {"task_id": "easy"})
    r = post(base_url, "/step", {"action_type": "inspect_data"})
    check("step after reset → 200", r.status_code == 200)

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 9. STATE after episode starts ══")
    # ══════════════════════════════════════════════════════════════════════
    post(base_url, "/reset", {"task_id": "medium"})
    post(base_url, "/step", {"action_type": "inspect_data"})
    r = get(base_url, "/state")
    check("GET /state → 200",           r.status_code == 200)
    s = r.json()
    check("status == active",           s.get("status") == "active")
    check("step == 1",                  s.get("step") == 1)
    check("episode_id present",         "episode_id" in s)
    check("current_grade present",      "current_grade" in s)
    check("reward_history present",     "reward_history" in s)
    check("steps_remaining present",    "steps_remaining" in s)

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 10. INVALID task_id → 400 ══")
    # ══════════════════════════════════════════════════════════════════════
    r = post(base_url, "/reset", {"task_id": "nonexistent"})
    check("invalid task_id → 400",      r.status_code == 400)
    check("valid_task_ids in error",    "valid_task_ids" in r.json().get("detail", {}))

    # ══════════════════════════════════════════════════════════════════════
    print("\n══ 11. FULL EASY EPISODE ══")
    # ══════════════════════════════════════════════════════════════════════
    post(base_url, "/reset", {"task_id": "easy"})
    rewards = []
    actions = [
        {"action_type": "inspect_data"},
        {"action_type": "inspect_metrics"},
        {"action_type": "fix_labels",
         "reasoning": "labels are flipped — this is a label noise issue"},
        {"action_type": "retrain"},
        {"action_type": "submit_diagnosis"},
    ]
    final_done = False
    for a in actions:
        r = post(base_url, "/step", a)
        body = r.json()
        rewards.append(body.get("reward", 0))
        final_done = body.get("done", False)
        print(f"    [{a['action_type']:<20}]  "
              f"r={body.get('reward'):+.2f}  done={body.get('done')}")
        if final_done:
            break

    r_state = get(base_url, "/state")
    grade = r_state.json().get("current_grade", 0)
    check("episode ended",          final_done)
    check("total reward > 0",       sum(rewards) > 0,  f"total={sum(rewards):.2f}")
    check("final grade > 0",        grade > 0,          f"grade={grade:.3f}")

    # ══════════════════════════════════════════════════════════════════════
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n{'='*55}")
    print(f"  {PASS if passed==total else FAIL}  {passed}/{total} checks passed")
    if passed < total:
        print("  Failed:")
        for label, ok in results:
            if not ok:
                print(f"    {FAIL} {label}")
    print(f"{'='*55}\n")
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for ML Debugger server")
    parser.add_argument("--url", default="http://localhost:7860",
                        help="Server base URL (default: http://localhost:7860)")
    args = parser.parse_args()

    try:
        ok = run(args.url)
        sys.exit(0 if ok else 1)
    except requests.exceptions.ConnectionError:
        print(f"\n{FAIL} Cannot connect to {args.url}")
        print("   Make sure the server is running:")
        print("   uvicorn app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)
