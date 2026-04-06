# tasks/task_hard.py
# Task 3 (Hard): Silent distribution shift.
# Train is StandardScaler-normalised. Test is raw (un-normalised).
# Model trains well (~85%+) but fails at inference (~52–58%).
# No single metric obviously reveals this — agent must compare distributions.
#
# Grader is 3-dimensional:
#   detection score  (0.3) — did agent call inspect_data and see the shift?
#   fix score        (0.4) — did agent apply fix_normalization to test set?
#   performance score(0.3) — does model achieve >80% on corrected test set?

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from data.generators import generate_hard_task_data, get_data_summary


VALID_ACTIONS = [
    "inspect_data", "inspect_metrics", "inspect_config",
    "fix_labels", "fix_normalization", "fix_learning_rate",
    "fix_architecture", "fix_loss_function", "fix_class_balance",
    "retrain", "submit_diagnosis"
]

REASONING_KEYWORDS = {
    "fix_normalization": [
        "distribution", "shift", "scale", "normaliz", "preprocess",
        "test", "train", "mismatch", "scaler", "std", "mean"
    ],
    "inspect_data": [
        "distribution", "shift", "mean", "std", "scale", "compare",
        "train", "test", "feature"
    ],
}


class HardTask:

    TASK_ID    = "hard"
    MAX_STEPS  = 15
    TOTAL_BUGS = 1   # one root cause: distribution shift

    def __init__(self):
        self.reset()

    def reset(self):
        (
            self.X_train_scaled,
            self.X_test_raw,
            self.X_test_scaled,
            self.y_train,
            self.y_test,
            self.scaler,
            self.shift_stats
        ) = generate_hard_task_data()

        # Agent always starts with raw (buggy) test set
        self.X_test_current = self.X_test_raw.copy()

        self.pipeline_state = {
            "learning_rate": 0.01,
            "max_iter":      500,
            "model_type":    "logistic_regression",
            "normalization": "applied_to_train_only",
        }

        # 3-dimensional grader tracking
        self.shift_detected   = False   # agent called inspect_data + saw signal
        self.fix_applied      = False   # agent applied fix_normalization to test
        self.prev_val_acc     = 0.0

        # Partial observability
        self.revealed = {
            "data":    False,
            "metrics": False,
            "config":  False,
        }

        self.step_count    = 0
        self.done          = False
        self.actions_taken = []
        self.last_3_val_accs = []

        # Train model on scaled train — it will perform well on train
        self.model = self._build_and_train()

        print(f"[ENV] Task HARD reset. Bug: silent distribution shift.")
        return self._build_observation(
            "Episode started. Model appears to train well but something may be wrong at inference."
        )

    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action):
        if self.done:
            return self._build_observation("Episode already finished."), 0.0, True, {}

        self.step_count += 1
        self.actions_taken.append(action.action_type)

        print(f"[ENV] Step {self.step_count}: action={action.action_type} param={action.parameter}")

        reward, result_msg = self._process_action(action)

        # Reasoning bonus
        bonus = self._check_reasoning_bonus(action)
        if bonus > 0:
            reward += bonus
            result_msg += f" [+{bonus:.2f} reasoning bonus]"

        if self._no_improvement():
            result_msg += " [No improvement in recent steps — try comparing train vs test distributions.]"

        if self.step_count >= self.MAX_STEPS:
            self.done = True
            result_msg += " [Max steps reached — episode ending.]"

        print(
            f"[ENV] Step {self.step_count} done. "
            f"reward={reward:.2f} detected={self.shift_detected} "
            f"fixed={self.fix_applied} done={self.done}"
        )

        obs  = self._build_observation(last_action_result=result_msg)
        info = {
            "shift_detected": self.shift_detected,
            "fix_applied":    self.fix_applied,
            "grade":          self.grade(),
        }
        return obs, reward, self.done, info

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION PROCESSOR
    # ──────────────────────────────────────────────────────────────────────────

    def _process_action(self, action):
        at = action.action_type
        reward = -0.05  # base step penalty

        if at not in VALID_ACTIONS:
            return -0.1, f"Unknown action '{at}'. Available: {VALID_ACTIONS}"

        # ── Inspect actions ───────────────────────────────────────────────────

        if at == "inspect_data":
            if not self.revealed["data"]:
                self.revealed["data"] = True
                reward = 0.1
                # Show train vs test distribution stats — the key signal
                # Toned down: shows numbers, doesn't say "distribution shift detected"
                s = self.shift_stats
                msg = (
                    f"Data inspection (train vs test comparison): "
                    f"train_mean={s['train_feature_mean']}, "
                    f"train_std={s['train_feature_std']}, "
                    f"test_mean={s['test_feature_mean']}, "
                    f"test_std={s['test_feature_std']}. "
                    f"Mean difference per feature: {s['mean_delta_per_feature']}. "
                    f"KS statistic on feature_0: {s['ks_stat_feature_0']} "
                    f"(higher = more divergence between train and test). "
                    f"Consider whether train and test data were prepared consistently."
                )
                # Mark shift as detected — contributes to grader score
                self.shift_detected = True
            else:
                reward = -0.1
                msg = "Already inspected data. No new information."

        elif at == "inspect_metrics":
            if not self.revealed["metrics"]:
                self.revealed["metrics"] = True
                reward = 0.1
                m = self._get_metrics()
                # The key signal: train acc is high, test acc is low
                # But we don't say WHY — agent must connect the dots
                msg = (
                    f"Metrics: train_acc={m['train_acc']:.3f}, "
                    f"val_acc={m['val_acc']:.3f}, "
                    f"train_loss={m['train_loss']:.4f}, "
                    f"val_loss={m['val_loss']:.4f}. "
                    f"There is a significant gap between training and validation performance. "
                    f"The model appears to generalise poorly despite good training metrics."
                )
            else:
                reward = -0.1
                msg = "Already inspected metrics. No new information."

        elif at == "inspect_config":
            if not self.revealed["config"]:
                self.revealed["config"] = True
                reward = 0.1
                msg = (
                    f"Config: {self.pipeline_state}. "
                    f"Hyperparameters look reasonable. "
                    f"Note: normalization is marked as 'applied_to_train_only' — "
                    f"verify whether the same preprocessing was applied to test data."
                )
            else:
                reward = -0.1
                msg = "Already inspected config. No new information."

        # ── Fix actions ───────────────────────────────────────────────────────

        elif at == "fix_normalization":
            if not self.fix_applied:
                self.fix_applied = True
                # Apply the same scaler (fitted on train) to test set
                self.X_test_current = self.scaler.transform(self.X_test_raw)
                self.pipeline_state["normalization"] = "applied_to_both"
                reward = 0.2
                new_mean = round(float(self.X_test_current.mean()), 4)
                new_std  = round(float(self.X_test_current.std()), 4)
                msg = (
                    f"Normalization fix applied: fitted scaler from training data "
                    f"applied to test set. "
                    f"Test feature mean={new_mean}, std={new_std}. "
                    f"Use 'retrain' or 'submit_diagnosis' to evaluate the effect."
                )
            else:
                reward = -0.1
                msg = "Normalization already fixed. No change."

        elif at == "retrain":
            reward = -0.05  # retrain penalty
            self.model = self._build_and_train()
            m = self._get_metrics()

            improved = m["val_acc"] > self.prev_val_acc + 0.01
            self.last_3_val_accs.append(m["val_acc"])
            if len(self.last_3_val_accs) > 3:
                self.last_3_val_accs.pop(0)

            if m["val_acc"] > 0.80 and self.fix_applied:
                reward = 0.5
                self.done = True
                msg = (
                    f"Retrain complete. train_acc={m['train_acc']:.3f}, "
                    f"val_acc={m['val_acc']:.3f}. "
                    f"Model now generalises well — TASK PASSED ✓"
                )
            elif improved:
                reward += 0.3
                msg = (
                    f"Retrain complete. train_acc={m['train_acc']:.3f}, "
                    f"val_acc={m['val_acc']:.3f}. "
                    f"Improvement detected (+{m['val_acc'] - self.prev_val_acc:.3f})."
                )
            else:
                reward += -0.1
                msg = (
                    f"Retrain complete. train_acc={m['train_acc']:.3f}, "
                    f"val_acc={m['val_acc']:.3f}. "
                    f"No meaningful improvement — root cause may not be addressed yet."
                )

            self.prev_val_acc = m["val_acc"]

        elif at == "submit_diagnosis":
            score = self.grade()
            self.done = True
            if score >= 0.8:
                reward = 0.5
                msg = f"Diagnosis submitted. Score: {score:.2f} — TASK PASSED ✓"
            elif score >= 0.4:
                reward = score * 0.4
                msg = f"Diagnosis submitted. Score: {score:.2f} — partial credit."
            else:
                reward = -0.1
                msg = f"Diagnosis submitted. Score: {score:.2f} — insufficient fix."

        elif at in ["fix_labels", "fix_learning_rate", "fix_class_balance",
                    "fix_architecture", "fix_loss_function"]:
            reward = -0.2
            msg = (
                f"'{at}' produced no improvement. "
                f"This does not appear to be the root cause."
            )

        else:
            reward = -0.1
            msg = f"No effect. Available actions: {VALID_ACTIONS}"

        return reward, msg

    # ──────────────────────────────────────────────────────────────────────────
    # GRADER — 3-dimensional
    # ──────────────────────────────────────────────────────────────────────────

    def grade(self):
        """
        FIX: performance component only activates when fix_applied=True.
        A model accidentally performing well on raw test data should NOT
        get free performance points — the fix must be explicitly applied.

        detection_score (0.3) — agent called inspect_data
        fix_score       (0.4) — agent applied fix_normalization
        perf_score      (0.3) — model accuracy on FIXED test set only
        """
        detection_score = 0.3 if self.shift_detected else 0.0
        fix_score       = 0.4 if self.fix_applied     else 0.0

        # Only evaluate perf on fixed test set — raw test perf doesn't count
        if self.fix_applied:
            y_pred    = self.model.predict(self.X_test_current)
            test_acc  = accuracy_score(self.y_test, y_pred)
            perf_norm = max(0.0, (test_acc - 0.50) / (0.95 - 0.50))
            perf_score = min(perf_norm, 1.0) * 0.3
        else:
            perf_score = 0.0

        return round(min(detection_score + fix_score + perf_score, 1.0), 4)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_and_train(self):
        model = LogisticRegression(
            C=1.0,
            max_iter=self.pipeline_state["max_iter"],
            random_state=42,
            solver="lbfgs"
        )
        # Always train on scaled train data
        model.fit(self.X_train_scaled, self.y_train)
        return model

    def _get_metrics(self):
        train_pred  = self.model.predict(self.X_train_scaled)
        val_pred    = self.model.predict(self.X_test_current)
        train_proba = self.model.predict_proba(self.X_train_scaled)
        val_proba   = self.model.predict_proba(self.X_test_current)
        return {
            "train_acc":  round(accuracy_score(self.y_train, train_pred), 4),
            "val_acc":    round(accuracy_score(self.y_test, val_pred), 4),
            "f1":         round(f1_score(self.y_test, val_pred, zero_division=0), 4),
            "train_loss": round(log_loss(self.y_train, train_proba), 4),
            "val_loss":   round(log_loss(self.y_test, val_proba), 4),
        }

    def _build_observation(self, last_action_result=""):
        # Always show train summary (agent needs this to compare with test)
        train_summary = get_data_summary(self.X_train_scaled, self.y_train)

        # Test summary only revealed after inspect_data
        if self.revealed["data"]:
            test_summary = get_data_summary(self.X_test_current, self.y_test)
        else:
            test_summary = {"note": "Run inspect_data to compare train vs test distributions."}

        if self.revealed["metrics"]:
            training_metrics = self._get_metrics()
        else:
            training_metrics = {"note": "Run inspect_metrics to see performance."}

        return {
            "step":               self.step_count,
            "task_id":            self.TASK_ID,
            "pipeline_state":     self.pipeline_state,
            "data_summary": {
                "train": train_summary,
                "test":  test_summary,
            },
            "training_metrics":   training_metrics,
            "last_action_result": last_action_result,
            "available_actions":  VALID_ACTIONS,
            "done":               self.done,
            "hint":               None,  # no hints on hard task
        }

    def _check_reasoning_bonus(self, action):
        if not action.reasoning:
            return 0.0
        keywords = REASONING_KEYWORDS.get(action.action_type, [])
        if any(kw in action.reasoning.lower() for kw in keywords):
            return 0.05
        return 0.0

    def _no_improvement(self):
        if len(self.last_3_val_accs) < 3:
            return False
        return max(self.last_3_val_accs) - min(self.last_3_val_accs) < 0.01
