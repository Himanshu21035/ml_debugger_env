# test_task_hard.py
from tasks.task_hard import HardTask
from models import Action

PASS, FAIL = "✅", "❌"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  →  {detail}" if detail else ""))
    results.append(condition)
    return condition


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 1. RESET ══")
# ══════════════════════════════════════════════════════════════════════════════
t = HardTask()
obs = t.reset()

check("step == 0",              obs["step"] == 0)
check("task_id == hard",        obs["task_id"] == "hard")
check("done == False",          obs["done"] == False)
check("hint is None",           obs["hint"] is None)
check("shift_detected = False", not t.shift_detected)
check("fix_applied = False",    not t.fix_applied)
check("prev_val_acc == 0.0",    t.prev_val_acc == 0.0)
check("initial grade == 0.0",   t.grade() == 0.0, f"grade={t.grade():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 2. INSPECT_DATA ══")
# ══════════════════════════════════════════════════════════════════════════════
t2 = HardTask(); t2.reset()
obs, r, done, info = t2.step(Action(action_type="inspect_data"))

check("reward == 0.1",          r == 0.1, f"r={r:.2f}")
check("done = False",           not done)
check("last_action_result set", len(obs["last_action_result"]) > 10)
check("diff in result",         "diff" in obs["last_action_result"].lower() or
                                 "mean" in obs["last_action_result"].lower())


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 3. INSPECT_METRICS ══")
# ══════════════════════════════════════════════════════════════════════════════
t3 = HardTask(); t3.reset()
obs, r, _, _ = t3.step(Action(action_type="inspect_metrics"))

check("reward == 0.1",        r == 0.1, f"r={r:.2f}")
check("train_acc in result",  "train" in obs["last_action_result"].lower())
check("gap in result",        "gap" in obs["last_action_result"].lower())


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 4. INSPECT_CONFIG ══")
# ══════════════════════════════════════════════════════════════════════════════
t4 = HardTask(); t4.reset()
obs, r, _, _ = t4.step(Action(action_type="inspect_config"))

check("reward == 0.1",               r == 0.1, f"r={r:.2f}")
check("normalization in result",
      "normalization" in obs["last_action_result"].lower() or
      "train_only"    in obs["last_action_result"].lower())


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 5. FIX_NORMALIZATION ══")
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
t5 = HardTask(); t5.reset()
obs, r, done, info = t5.step(Action(action_type="fix_normalization"))

check("reward == 0.2",          r == 0.2, f"r={r:.2f}")
check("fix_applied = True",     t5.fix_applied)
check("done = False",           not done)
check("test std ≈ 1 after fix",
      abs(t5.X_test_current.std() - 1.0) < 0.3,
      f"std={t5.X_test_current.std():.3f}")
check("grade = 0.0 still (no retrain yet, no detection)",
      t5.grade() == 0.0, f"grade={t5.grade():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 6. RETRAIN WITHOUT FIX → penalty ══")
# ══════════════════════════════════════════════════════════════════════════════
t6 = HardTask(); t6.reset()
_, r, _, _ = t6.step(Action(action_type="retrain"))
check("retrain without fix → -0.1", r <= -0.1, f"r={r:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 7. RETRAIN AFTER FIX ══")
# ══════════════════════════════════════════════════════════════════════════════
t7 = HardTask(); t7.reset()
t7.step(Action(action_type="fix_normalization"))
obs, r, done, info = t7.step(Action(action_type="retrain"))

check("retrain reward >= 0.1",    r >= 0.1, f"r={r:.2f}")
check("retrained = True",         t7.retrained)
check("test_acc_after_fix set",   t7.test_acc_after_fix is not None)
check("test_acc > 0.75 after fix",
      t7._get_metrics()["test_acc"] > 0.75,
      f"test_acc={t7._get_metrics()['test_acc']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 8. SUBMIT_DIAGNOSIS — correct reasoning ══")
# ══════════════════════════════════════════════════════════════════════════════
t8 = HardTask(); t8.reset()
obs, r, done, info = t8.step(Action(
    action_type="submit_diagnosis",
    reasoning="distribution shift detected between train and test"
))
check("reward == 0.3",          r == 0.3,  f"r={r:.2f}")
check("shift_detected = True",  t8.shift_detected)
check("done = True",            done)

# ── wrong reasoning
t8b = HardTask(); t8b.reset()
_, r2, done2, _ = t8b.step(Action(
    action_type="submit_diagnosis",
    reasoning="I think the model is too simple"
))
check("wrong reasoning → -0.1", r2 <= -0.1, f"r={r2:.2f}")
check("done = True even on wrong diagnosis", done2)


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 9. WRONG ACTIONS → -0.2 ══")
# ══════════════════════════════════════════════════════════════════════════════
t9 = HardTask(); t9.reset()
for bad in ["fix_labels", "fix_learning_rate", "fix_class_balance"]:
    _, r, _, _ = t9.step(Action(action_type=bad))
    check(f"{bad} → -0.2", r <= -0.2, f"r={r:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 10. INVALID ACTION → -0.1 ══")
# ══════════════════════════════════════════════════════════════════════════════
t10 = HardTask(); t10.reset()
obs, r, _, _ = t10.step(Action(action_type="totally_invalid"))
check("invalid → -0.1",             r <= -0.1, f"r={r:.2f}")
check("Available in result",
      "Available" in obs["last_action_result"])


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 11. GRADE LADDER ══")
# ══════════════════════════════════════════════════════════════════════════════
t11 = HardTask(); t11.reset()
g0 = t11.grade()
check("grade=0 at start", g0 == 0.0, f"grade={g0:.3f}")

t11.shift_detected = True
g1 = t11.grade()
check("grade=0.3 after detection only", abs(g1 - 0.3) < 0.01, f"grade={g1:.3f}")

t11.step(Action(action_type="fix_normalization"))
t11.step(Action(action_type="retrain"))
g2 = t11.grade()
check("grade=0.7 after detect+fix+retrain", g2 >= 0.7, f"grade={g2:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 12. FULL OPTIMAL EPISODE ══")
# ══════════════════════════════════════════════════════════════════════════════
t12 = HardTask(); t12.reset()
rewards = []

steps = [
    Action(action_type="inspect_data"),
    Action(action_type="inspect_metrics"),
    Action(action_type="inspect_config"),
    Action(action_type="fix_normalization"),
    Action(action_type="retrain"),
    Action(action_type="submit_diagnosis",
           reasoning="distribution shift — train normalized, test raw"),
]

for a in steps:
    obs, r, done, info = t12.step(a)
    rewards.append(r)
    print(f"    [{a.action_type:<20}]  r={r:+.2f}  "
          f"detected={info['shift_detected']}  fixed={info['fix_applied']}  "
          f"grade={info['grade']:.3f}  done={done}")
    if done:
        break

check("shift detected",        info["shift_detected"])
check("fix applied",           info["fix_applied"])
check("final grade >= 0.7",    info["grade"] >= 0.7, f"grade={info['grade']:.3f}")
check("episode ended",         done)
check("total reward > 0",      sum(rewards) > 0, f"total={sum(rewards):.2f}")


# ══════════════════════════════════════════════════════════════════════════════
passed = sum(results)
total  = len(results)
print(f"\n{'='*55}")
print(f"  {PASS if passed==total else FAIL}  {passed}/{total} checks passed")
if passed < total:
    print(f"  {total-passed} failed — fix before submitting!")
print(f"{'='*55}\n")
