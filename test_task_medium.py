# test_task_medium.py

import numpy as np
from tasks.task_medium import MediumTask
from models import Action

PASS = "✅"
FAIL = "❌"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  →  {detail}" if detail else ""))
    return condition


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ RESET ══")
# ══════════════════════════════════════════════════════════════════════════════
task = MediumTask()
obs  = task.reset()

check("step == 0",           obs["step"] == 0)
check("task_id == medium",   obs["task_id"] == "medium")
check("done == False",       obs["done"] == False)
check("hint present",        obs["hint"] is not None)
check("no bugs fixed",       sum(task.bugs_fixed.values()) == 0)
check("prev_val_acc == 0",   task.prev_val_acc == 0.0)

# Partial observability — locked at reset
check("data hidden at start",    "note" in obs["data_summary"])
check("metrics hidden at start", "note" in obs["training_metrics"])
check("lr hidden at start",      obs["pipeline_state"]["learning_rate"] == "??? (run inspect_config to reveal)")

initial_grade = task.grade()
check("initial grade < 0.15 (bugs active, F1≈0)", initial_grade < 0.15,
      f"grade={initial_grade:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ INSPECT ACTIONS (unlock signals) ══")
# ══════════════════════════════════════════════════════════════════════════════
task2 = MediumTask()
task2.reset()

obs, r, done, _ = task2.step(Action(action_type="inspect_data", reasoning="checking class distribution"))
check("inspect_data reward == 0.1", abs(r - 0.1) < 0.01, f"r={r:.2f}")
check("data revealed after inspect", "class_distribution" in obs["data_summary"])
check("done still False", not done)

obs, r, done, _ = task2.step(Action(action_type="inspect_data"))
check("repeat inspect penalised", r <= -0.1, f"r={r:.2f}")

obs, r, done, _ = task2.step(Action(action_type="inspect_metrics"))
check("inspect_metrics reward == 0.1", abs(r - 0.1) < 0.01, f"r={r:.2f}")
check("metrics revealed after inspect", "val_acc" in obs["training_metrics"])

obs, r, done, _ = task2.step(Action(action_type="inspect_config"))
check("inspect_config reward == 0.1", abs(r - 0.1) < 0.01, f"r={r:.2f}")
check("lr revealed after inspect", isinstance(obs["pipeline_state"]["learning_rate"], float))


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ FIX ACTIONS ══")
# ══════════════════════════════════════════════════════════════════════════════
task3 = MediumTask()
task3.reset()

# Fix class balance
obs, r, done, info = task3.step(Action(
    action_type="fix_class_balance",
    reasoning="class imbalance is causing poor minority recall"
))
check("fix_class_balance reward >= 0.2", r >= 0.2, f"r={r:.2f}")
check("reasoning bonus applied",  r > 0.2, f"r={r:.2f} (should be 0.25)")
check("class_balance bug marked fixed", task3.bugs_fixed["class_balance"])
check("pipeline_state updated", task3.pipeline_state["class_weight"] == "balanced")

# Repeat fix — should penalise
obs, r, done, _ = task3.step(Action(action_type="fix_class_balance"))
check("repeat fix penalised", r <= -0.1, f"r={r:.2f}")

# Fix normalization
obs, r, done, _ = task3.step(Action(
    action_type="fix_normalization",
    reasoning="feature magnitude is too large, need to normalize scale"
))
check("fix_normalization reward >= 0.2", r >= 0.2, f"r={r:.2f}")
check("normalization bug marked fixed", task3.bugs_fixed["normalization"])
# After normalization, feature std should be closer to 1
new_std = task3.X_train_current.std()
check("features rescaled after fix", abs(new_std - 1.0) < 0.5, f"std={new_std:.3f}")

# Fix learning rate
obs, r, done, _ = task3.step(Action(
    action_type="fix_learning_rate",
    parameter="0.01",
    reasoning="learning rate is too high, causes unstable convergence"
))
check("fix_learning_rate reward >= 0.2", r >= 0.2, f"r={r:.2f}")
check("lr bug marked fixed", task3.bugs_fixed["learning_rate"])
check("lr updated in pipeline_state", task3.pipeline_state["learning_rate"] == 0.01)


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ RETRAIN + REWARD IMPROVEMENT ══")
# ══════════════════════════════════════════════════════════════════════════════

# Retrain with all 3 fixes applied
pre_retrain_grade = task3.grade()
obs, r, done, info = task3.step(Action(action_type="retrain"))
post_retrain_grade = task3.grade()

check("retrain has step cost",  r != 0.5 or done,  f"r={r:.2f}")  # not free
check("grade improved after all fixes", post_retrain_grade > pre_retrain_grade,
      f"{pre_retrain_grade:.4f} → {post_retrain_grade:.4f}")
check("prev_val_acc updated after retrain", task3.prev_val_acc > 0.0,
      f"prev_val_acc={task3.prev_val_acc:.4f}")
check("bugs_fixed info correct", info["bugs_fixed"] == 3, str(info))


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ FULL OPTIMAL EPISODE ══")
# ══════════════════════════════════════════════════════════════════════════════
task4   = MediumTask()
task4.reset()
rewards = []

optimal_steps = [
    Action(action_type="inspect_data",     reasoning="checking for class imbalance"),
    Action(action_type="inspect_metrics",  reasoning="checking loss divergence"),
    Action(action_type="inspect_config",   reasoning="checking learning rate"),
    Action(action_type="fix_class_balance",reasoning="class imbalance is causing poor minority recall"),
    Action(action_type="fix_normalization",reasoning="feature magnitude is too large, need to normalize scale"),
    Action(action_type="fix_learning_rate",parameter="0.01",
           reasoning="learning rate is too high, causes unstable convergence"),
    Action(action_type="retrain"),
    Action(action_type="submit_diagnosis"),
]

for a in optimal_steps:
    obs, r, done, info = task4.step(a)
    rewards.append(r)
    print(f"    [{a.action_type}]  r={r:.2f}  bugs={info['bugs_fixed']}/3  done={done}")
    if done:
        break

final_grade = task4.grade()
total_reward = sum(rewards)

check("All 3 bugs fixed",          info["bugs_fixed"] == 3,    str(info["bugs_fixed"]))
check("Final grade > 0.55",         final_grade > 0.55,          f"grade={final_grade:.4f}")
check("Episode ended (done=True)", done)
check("Total reward > 0",          total_reward > 0,           f"total={total_reward:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ WRONG ACTIONS PENALISED ══")
# ══════════════════════════════════════════════════════════════════════════════
task5 = MediumTask()
task5.reset()

obs, r, _, _ = task5.step(Action(action_type="fix_labels"))
check("fix_labels penalised on medium task", r <= -0.2, f"r={r:.2f}")

obs, r, _, _ = task5.step(Action(action_type="fix_architecture"))
check("fix_architecture penalised", r <= -0.2, f"r={r:.2f}")

obs, r, _, _ = task5.step(Action(action_type="nonsense_action"))
check("invalid action returns -0.1", r <= -0.1, f"r={r:.2f}")
check("invalid action message has valid list", "Available actions" in obs["last_action_result"])


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ GRADE CALIBRATION ══")
# ══════════════════════════════════════════════════════════════════════════════
task6 = MediumTask()
task6.reset()
# ══ GRADE CALIBRATION — update thresholds ══
no_fix_grade = task6.grade()
check("No-fix grade < 0.15 (F1≈0 with imbalance)",
      no_fix_grade < 0.15, f"grade={no_fix_grade:.4f}")

# Partial fix (1 bug)
task6.step(Action(action_type="fix_class_balance", reasoning="class imbalance"))
task6.step(Action(action_type="retrain"))
one_fix_grade = task6.grade()
check("1-bug grade > no-fix grade",
      one_fix_grade > no_fix_grade,
      f"{no_fix_grade:.4f} → {one_fix_grade:.4f}")

print(f"\n  Grade progression: none={no_fix_grade:.4f} → 1_fix={one_fix_grade:.4f} → all_fixed={final_grade:.4f}")
