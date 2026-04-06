# test_generators.py
# Run this after any change to data/generators.py to verify all 3 tasks.

import numpy as np
from data.generators import (
    generate_easy_task_data,
    generate_medium_task_data,
    generate_hard_task_data,
    get_data_summary,
    get_baseline_scores,
)

PASS = "✅"
FAIL = "❌"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  →  {detail}" if detail else ""))
    return condition


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ TASK 1 — Easy (Label Flipping) ══")
# ══════════════════════════════════════════════════════════════════════════════

X_train, X_test, y_clean, y_buggy, y_test = generate_easy_task_data()

flip_rate = (y_clean != y_buggy).mean()
unique, counts = np.unique(y_buggy, return_counts=True)

check("X_train shape",       X_train.shape == (800, 10),      str(X_train.shape))
check("X_test shape",        X_test.shape  == (200, 10),      str(X_test.shape))
check("y_clean shape",       y_clean.shape == (800,),         str(y_clean.shape))
check("Label flip ~30%",     0.28 <= flip_rate <= 0.32,       f"{flip_rate:.2%}")
check("clean != buggy",      not np.array_equal(y_clean, y_buggy), "labels differ")
check("No nulls in X_train", np.isnan(X_train).sum() == 0,   "0 nulls")

summary = get_data_summary(X_train, y_buggy)
check("data_summary has skewness",     "skewness" in summary,            str(summary["skewness"]))
check("data_summary has kurtosis",     "kurtosis" in summary,            str(summary["kurtosis"]))
check("data_summary has top_features", "top_extreme_features" in summary, "present")
check("data_summary minority_fraction","minority_fraction" in summary,    str(summary["minority_fraction"]))


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ TASK 2 — Medium (3 Bugs) ══")
# ══════════════════════════════════════════════════════════════════════════════

X_tr_buggy, X_tr_clean, X_te, y_tr, y_te, cfg_buggy, cfg_clean = generate_medium_task_data()

minority_frac = (y_tr == 1).mean()
buggy_first5_mean  = abs(X_tr_buggy[:, :5].mean())
clean_first5_mean  = abs(X_tr_clean[:, :5].mean()) + 1e-9
buggy_last10_mean  = abs(X_tr_buggy[:, 5:].mean())
clean_last10_mean  = abs(X_tr_clean[:, 5:].mean()) + 1e-9

check("X_train_buggy shape",     X_tr_buggy.shape == (800, 15),    str(X_tr_buggy.shape))
check("X_train_clean shape",     X_tr_clean.shape == (800, 15),    str(X_tr_clean.shape))
check("X_test shape",            X_te.shape == (200, 15),          str(X_te.shape))
check("Class imbalance <15%",    minority_frac <= 0.15,             f"minority={minority_frac:.2%}")
check("First 5 features over-scaled", buggy_first5_mean / clean_first5_mean > 10,
      f"ratio={buggy_first5_mean / clean_first5_mean:.1f}×")
check("Last 10 features NOT over-scaled", buggy_last10_mean / clean_last10_mean < 5,
      f"ratio={buggy_last10_mean / clean_last10_mean:.1f}×")
check("Buggy LR is 1.0",         cfg_buggy["learning_rate"] == 1.0,  str(cfg_buggy["learning_rate"]))
check("Clean LR is 0.01",        cfg_clean["learning_rate"] == 0.01, str(cfg_clean["learning_rate"]))
check("Buggy class_weight=None", cfg_buggy["class_weight"] is None,  str(cfg_buggy["class_weight"]))
check("Clean class_weight=balanced", cfg_clean["class_weight"] == "balanced", "balanced")
check("Buggy != clean features", not np.allclose(X_tr_buggy, X_tr_clean), "differ")
check("Test set is pre-scaled",  abs(X_te.mean()) < 1.0,             f"mean={X_te.mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ TASK 3 — Hard (Distribution Shift) ══")
# ══════════════════════════════════════════════════════════════════════════════

X_tr_sc, X_te_raw, X_te_sc, y_tr, y_te, scaler, shift_stats = generate_hard_task_data()

check("X_train_scaled shape",    X_tr_sc.shape == (900, 20),   str(X_tr_sc.shape))
check("X_test_raw shape",        X_te_raw.shape == (300, 20),  str(X_te_raw.shape))
check("X_test_scaled shape",     X_te_sc.shape == (300, 20),   str(X_te_sc.shape))

check("Train mean ≈ 0",          abs(X_tr_sc.mean()) < 0.05,   f"mean={X_tr_sc.mean():.4f}")
check("Train std ≈ 1",           abs(X_tr_sc.std() - 1.0) < 0.05, f"std={X_tr_sc.std():.4f}")
check("Test raw mean ≠ 0",       abs(X_te_raw.mean()) > 0.01,  f"mean={X_te_raw.mean():.4f}")
check("Test raw std ≠ 1",        abs(X_te_raw.std() - 1.0) > 0.2, f"std={X_te_raw.std():.4f}")
check("Test scaled mean ≈ 0",    abs(X_te_sc.mean()) < 0.1,    f"mean={X_te_sc.mean():.4f}")
check("Test scaled std ≈ 1",     abs(X_te_sc.std() - 1.0) < 0.1, f"std={X_te_sc.std():.4f}")

check("shift_stats has ks_stat",          "ks_stat_feature_0" in shift_stats,
      str(shift_stats.get("ks_stat_feature_0")))
check("shift_stats has mean_delta",       "mean_delta_per_feature" in shift_stats,
      str(shift_stats.get("mean_delta_per_feature")))
check("shift_stats has std_delta",        "std_delta_per_feature" in shift_stats,
      str(shift_stats.get("std_delta_per_feature")))
check("shift_detected is True",           shift_stats["shift_detected"] is True, "True")
check("fix_method present",               "fix_method" in shift_stats,
      shift_stats.get("fix_method", "")[:40])
check("KS stat > 0.1 (shift is real)",    shift_stats["ks_stat_feature_0"] > 0.1,
      f"ks={shift_stats['ks_stat_feature_0']:.4f}")

# Verify scaler can transform test correctly
X_te_refitted = scaler.transform(X_te_raw)
check("Scaler re-transform matches X_test_scaled",
      np.allclose(X_te_refitted, X_te_sc, atol=1e-6), "arrays match")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ UTILITY — get_baseline_scores() ══")
# ══════════════════════════════════════════════════════════════════════════════

baseline = get_baseline_scores()
for task_name, scores in baseline.items():
    check(f"{task_name} has min/max/note",
          all(k in scores for k in ["min", "max", "note"]),
          f"min={scores['min']} max={scores['max']}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ DETERMINISM CHECK — run generators twice, compare ══")
# ══════════════════════════════════════════════════════════════════════════════

X1, _, y1_clean, y1_buggy, _ = generate_easy_task_data()
X2, _, y2_clean, y2_buggy, _ = generate_easy_task_data()
check("Easy task is deterministic", np.array_equal(X1, X2) and np.array_equal(y1_buggy, y2_buggy))

_, _, X_te_a, y_tr_a, _, _, _ = generate_medium_task_data()
_, _, X_te_b, y_tr_b, _, _, _ = generate_medium_task_data()
check("Medium task is deterministic", np.array_equal(X_te_a, X_te_b) and np.array_equal(y_tr_a, y_tr_b))

X_s1, _, _, _, _, _, ss1 = generate_hard_task_data()
X_s2, _, _, _, _, _, ss2 = generate_hard_task_data()
check("Hard task is deterministic",   np.array_equal(X_s1, X_s2) and ss1 == ss2)


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ SUMMARY ══")
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nShift stats (Task 3 inspect signal):")
for k, v in shift_stats.items():
    print(f"  {k}: {v}")

print(f"\nBaseline scores:")
for task, s in baseline.items():
    print(f"  {task}: {s['min']}–{s['max']}  ({s['note']})")
