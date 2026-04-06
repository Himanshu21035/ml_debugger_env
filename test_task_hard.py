# test_task_hard.py

from tasks.task_hard import HardTask
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
task = HardTask()
obs  = task.reset()

check("step == 0",           obs["step"] == 0)
check("task_id == hard",     obs["task_id"] == "hard")
check("done == False",       obs["done"] == False)
check("no hint on hard",     obs["hint"] is None)
check("shift_detected=False", not task.shift_detected)
check("fix_applied=False",   not task.fix_applied)
check("prev_val_acc == 0",   task.prev_val_acc == 0.0)

# Partial observability
check("data locked at reset",    "note" in obs["data_summary"]["test"])
check("metrics locked at reset", "note" in obs["training_metrics"])

# Initial grade: no detection, no fix, poor test perf → should be near 0
initial_grade = task.grade()
check("initial grade < 0.15 (no detection/fix)", initial_grade < 0.15,
      f"grade={initial_grade:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ INSPECT_DATA — reveals shift signal + marks detection ══")
# ══════════════════════════════════════════════════════════════════════════════
task2 = HardTask()
task2.reset()

obs, r, done, info = task2.step(Action(
    action_type="inspect_data",
    reasoning="comparing train and test distribution to check for shift"
))
check("inspect_data reward >= 0.1",  r >= 0.1, f"r={r:.2f}")
check("reasoning bonus applied",     r > 0.1,              f"r={r:.2f}")
check("shift_detected = True",       task2.shift_detected)
check("test summary unlocked",       "feature_mean_global" in obs["data_summary"]["test"])
check("shift stats in message",      "KS" in obs["last_action_result"])
check("grade now has detection",     task2.grade() >= 0.3, f"grade={task2.grade():.4f}")

# Repeat inspect — should penalise
obs, r, _, _ = task2.step(Action(action_type="inspect_data"))
check("repeat inspect penalised", r <= -0.1, f"r={r:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ INSPECT_METRICS — reveals train/val gap ══")
# ══════════════════════════════════════════════════════════════════════════════
task3 = HardTask()
task3.reset()

obs, r, _, _ = task3.step(Action(action_type="inspect_metrics"))
check("inspect_metrics reward >= 0.1", r >= 0.1, f"r={r:.2f}")
check("metrics unlocked",   "train_acc" in obs["training_metrics"])
check("train_acc > val_acc (shift visible)",
      obs["training_metrics"]["train_acc"] > obs["training_metrics"]["val_acc"],
      f"train={obs['training_metrics']['train_acc']:.3f} val={obs['training_metrics']['val_acc']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ INSPECT_CONFIG — hints at normalization mismatch ══")
# ══════════════════════════════════════════════════════════════════════════════
task4 = HardTask()
task4.reset()

obs, r, _, _ = task4.step(Action(action_type="inspect_config"))
check("inspect_config reward >= 0.1", r >= 0.1, f"r={r:.2f}")
check("normalization hint in message",
      "applied_to_train_only" in obs["last_action_result"] or
      "preprocessing" in obs["last_action_result"].lower())


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ FIX_NORMALIZATION ══")
# ══════════════════════════════════════════════════════════════════════════════
task5 = HardTask()
task5.reset()

import numpy as np
pre_fix_test_std = task5.X_test_current.std()

obs, r, done, info = task5.step(Action(
    action_type="fix_normalization",
    reasoning="distribution shift detected — applying scaler to test set"
))
check("fix_normalization reward >= 0.2", r >= 0.2, f"r={r:.2f}")
check("reasoning bonus applied", r > 0.2, f"r={r:.2f}")
check("fix_applied = True", task5.fix_applied)
check("test std ≈ 1 after fix", abs(task5.X_test_current.std() - 1.0) < 0.2,
      f"std={task5.X_test_current.std():.3f}")
check("pipeline normalization updated",
      task5.pipeline_state["normalization"] == "applied_to_both")
check("grade has fix component",
      task5.grade() >= 0.4, f"grade={task5.grade():.4f}")

# Repeat fix — should penalise
obs, r, _, _ = task5.step(Action(action_type="fix_normalization"))
check("repeat fix penalised", r <= -0.1, f"r={r:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ RETRAIN AFTER FIX ══")
# ══════════════════════════════════════════════════════════════════════════════
task6 = HardTask()
task6.reset()
task6.step(Action(action_type="fix_normalization",
                  reasoning="distribution shift, applying scaler to test"))
obs, r, done, info = task6.step(Action(action_type="retrain"))

check("retrain after fix passes threshold", done or r >= 0.3, f"r={r:.2f} done={done}")
check("val_acc > 0.80 after fix+retrain",
      task6._get_metrics()["val_acc"] > 0.80,
      f"val_acc={task6._get_metrics()['val_acc']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ FULL OPTIMAL EPISODE ══")
# ══════════════════════════════════════════════════════════════════════════════
task7   = HardTask()
task7.reset()
rewards = []

optimal = [
    Action(action_type="inspect_data",
           reasoning="comparing train and test distribution to check for shift"),
    Action(action_type="inspect_metrics",
           reasoning="checking train vs test performance gap"),
    Action(action_type="inspect_config",
           reasoning="checking if preprocessing was applied consistently"),
    Action(action_type="fix_normalization",
           reasoning="distribution shift detected — applying scaler to test set"),
    Action(action_type="retrain"),
    Action(action_type="submit_diagnosis"),
]

for a in optimal:
    obs, r, done, info = task7.step(a)
    rewards.append(r)
    print(f"    [{a.action_type}]  r={r:.2f}  detected={info['shift_detected']}  "
          f"fixed={info['fix_applied']}  grade={info['grade']:.4f}  done={done}")
    if done:
        break

final_grade  = task7.grade()
total_reward = sum(rewards)

check("Shift detected",         info["shift_detected"])
check("Fix applied",            info["fix_applied"])
check("Final grade >= 0.7",     final_grade >= 0.7,  f"grade={final_grade:.4f}")
check("Episode ended",          done)
check("Total reward > 0",       total_reward > 0,    f"total={total_reward:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ WRONG ACTIONS PENALISED ══")
# ══════════════════════════════════════════════════════════════════════════════
task8 = HardTask()
task8.reset()

for bad_action in ["fix_labels", "fix_learning_rate", "fix_class_balance"]:
    obs, r, _, _ = task8.step(Action(action_type=bad_action))
    check(f"{bad_action} penalised", r <= -0.2, f"r={r:.2f}")

obs, r, _, _ = task8.step(Action(action_type="totally_invalid"))
check("invalid action returns -0.1", r <= -0.1, f"r={r:.2f}")
check("invalid action lists valid actions", "Available" in obs["last_action_result"])


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ GRADE BREAKDOWN ══")
# ══════════════════════════════════════════════════════════════════════════════
task9 = HardTask()
task9.reset()
g0 = task9.grade()
check("grade=0 before anything",   g0 == 0.0,   f"grade={g0:.4f}") 

task9.shift_detected = True
g1 = task9.grade()
check("grade=0.3 after detection only", abs(g1 - 0.3) < 0.02, f"grade={g1:.4f}")

task9.step(Action(action_type="fix_normalization",
                  reasoning="distribution mismatch between train and test"))
g2 = task9.grade()
check("grade≥0.7 after detection+fix", g2 >= 0.7, f"grade={g2:.4f}")


print(f"\n  Grade ladder: 0→{g0:.4f} → detect→{g1:.4f} → fix→{g2:.4f} → full→{final_grade:.4f}")
