# tasks/task_easy.py
# Task 1 (Easy): Single bug — 30% of training labels are flipped.
# Agent must identify it via inspect actions and apply fix_labels.
# Success = model accuracy > 85% after retraining on fixed labels.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from data.generators import generate_easy_task_data, get_data_summary


VALID_ACTIONS = [
    "inspect_data", "inspect_metrics", "inspect_config", "inspect_model",
    "fix_labels", "fix_normalization", "fix_learning_rate",
    "fix_architecture", "fix_loss_function", "fix_class_balance",
    "retrain", "submit_diagnosis", 
]

# Reasoning bonus keywords for correct diagnosis
REASONING_KEYWORDS = {
    "fix_labels": ["label", "flip", "noise", "corrupt", "mislabel"],
}
# Add this helper at the top of each task file (or in a shared utils):
def _clamp_grade(score: float) -> float:
    """Validator requires strictly (0, 1) — not 0.0, not 1.0."""
    return round(max(0.001, min(score, 0.999)), 4)

class EasyTask:

    TASK_ID = "easy"
    MAX_STEPS = 15
    BUG_CATEGORY = "bad_data"

    def __init__(self):
        self.reset()

    def reset(self):
        (
            self.X_train,
            self.X_test,
            self.y_train_clean,
            self.y_train_buggy,
            self.y_test
        ) = generate_easy_task_data()

        self.y_train_current = self.y_train_buggy.copy()

        self.pipeline_state = {
            "learning_rate": 0.01,
            "max_iter": 500,
            "model_type": "logistic_regression",
            "normalization": "none"
        }

        self.model = self._build_and_train()

        # Episode tracking
        self.step_count = 0
        self.done = False
        self.label_fix_applied = False
        self.retrained_after_fix = False
        self.last_inspected = set()
        self.actions_taken = []
        # self.last_3_rewards = []     # for early termination check

        print(f"[ENV] Task EASY reset. Bug injected: label_flip (30%)")

        return self._build_observation(
            last_action_result="Episode started. Model trained on current data.",
            hint="Model performance is surprisingly poor. Something in the training data may be off."
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
            result_msg += f" [+{bonus} reasoning bonus]"

        # # Track rewards for early termination
        # self.last_3_rewards.append(reward)
        # if len(self.last_3_rewards) > 3:
        #     self.last_3_rewards.pop(0)

        # Early termination: 3 consecutive useless/bad actions
        # if self._stuck_detected():
        #     self.done = True
        #     result_msg += " [No progress detected in last 3 steps — episode ending early.]"

        # Max steps
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            result_msg += " [Max steps reached — episode ending.]"

        print(f"[ENV] Step {self.step_count} done. reward={reward:.2f} done={self.done}")

        obs = self._build_observation(last_action_result=result_msg)
        info = {"bugs_fixed": int(self.label_fix_applied and self.retrained_after_fix)}

        return obs, reward, self.done, info

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION PROCESSOR
    # ──────────────────────────────────────────────────────────────────────────

    def _process_action(self, action):
        at = action.action_type
        reward = -0.05  # base step penalty

        # ── Validate action ──────────────────────────────────────────────────
        if at not in VALID_ACTIONS:
            return -0.1, (
                f"Invalid action '{at}'. "
                f"Available actions: {VALID_ACTIONS}"
            )

        # ── Inspect actions ──────────────────────────────────────────────────

        if at == "inspect_data":
            if "inspect_data" not in self.last_inspected:
                self.last_inspected.add("inspect_data")
                reward = 0.1
                unique, counts = np.unique(self.y_train_current, return_counts=True)
                class_dist = dict(zip(unique.tolist(), counts.tolist()))
                msg = (
                    f"Data inspection: {len(self.y_train_current)} training samples. "
                    f"Class distribution: {class_dist}. "
                    f"Label consistency appears low — consider investigating data quality."
                )
            else:
                reward = -0.1
                msg = "Already inspected data. No new information gained."

        elif at == "inspect_metrics":
            if "inspect_metrics" not in self.last_inspected:
                self.last_inspected.add("inspect_metrics")
                reward = 0.1
                m = self._get_metrics()
                msg = (
                    f"Metrics: train_acc={m['train_acc']:.3f}, val_acc={m['val_acc']:.3f}, "
                    f"train_loss={m['train_loss']:.4f}, val_loss={m['val_loss']:.4f}. "
                    f"Both train and val accuracy are near random — "
                    f"this pattern may suggest an issue in the training data itself."
                )
            else:
                reward = -0.1
                msg = "Already inspected metrics. No new information gained."

        elif at == "inspect_config":
            if "inspect_config" not in self.last_inspected:
                self.last_inspected.add("inspect_config")
                reward = 0.1
                msg = (
                    f"Config: {self.pipeline_state}. "
                    f"Hyperparameters appear within normal range. "
                    f"Architecture looks appropriate for this dataset size."
                )
            else:
                reward = -0.1
                msg = "Already inspected config. No new information gained."
        
        elif at == "inspect_model":
            if "inspect_model" not in self.last_inspected:
                self.last_inspected.add("inspect_model")
                reward = 0.1
                try:
                    proba   = self.model.predict_proba(self.X_test)
                    conf    = proba.max(axis=1).mean()
                    low_conf = (proba.max(axis=1) < 0.6).mean()
                    coef_norm = float(np.linalg.norm(self.model.coef_))
                except Exception:
                    conf, low_conf, coef_norm = 0.0, 0.0, 0.0
                msg = (
                    f"Model inspection: avg_confidence={conf:.3f}, "
                    f"low_confidence_ratio={low_conf:.3f}, "
                    f"coef_norm={coef_norm:.3f}. "
                    f"{'Low confidence suggests model is uncertain — root cause not fixed yet.' if conf < 0.65 else 'Confidence looks healthy.'}"
                )
            else:
                reward = -0.1
                msg = "Already inspected model. No new information."
        # ── Fix actions ──────────────────────────────────────────────────────

        elif at == "fix_labels":
            if not self.label_fix_applied:
                self.y_train_current = self.y_train_clean.copy()
                self.label_fix_applied = True
                reward = 0.2
                msg = (
                    "Label fix applied: training labels have been restored. "
                    "Use 'retrain' to measure the effect."
                )
            else:
                reward = -0.1
                msg = "Labels already fixed. No change applied."

        elif at == "retrain":
            # FIX: extra retrain penalty to discourage spamming
            reward = -0.05
            self.model = self._build_and_train()
            if self.label_fix_applied:
                self.retrained_after_fix = True
            m = self._get_metrics()

            if m["val_acc"] > 0.80:
                reward = 0.5
                self.done = True
                msg = (
                    f"Retrain complete. train_acc={m['train_acc']:.3f}, "
                    f"val_acc={m['val_acc']:.3f}, val_loss={m['val_loss']:.4f}. "
                    f"SUCCESS — model exceeds 80% accuracy threshold!"
                )
            elif m["val_acc"] > 0.65:
                reward = 0.3
                msg = (
                    f"Retrain complete. train_acc={m['train_acc']:.3f}, "
                    f"val_acc={m['val_acc']:.3f}, val_loss={m['val_loss']:.4f}. "
                    f"Improvement detected but threshold not yet reached."
                )
            else:
                reward = -0.2
                msg = (
                    f"Retrain complete. train_acc={m['train_acc']:.3f}, "
                    f"val_acc={m['val_acc']:.3f}, val_loss={m['val_loss']:.4f}. "
                    f"No significant improvement. The root cause may not have been addressed."
                )

        elif at == "submit_diagnosis":
            score = self.grade()
            self.done = True
            if score >= 0.8:
                reward = 0.5
                msg = f"Diagnosis submitted. Grader score: {score:.2f} — TASK PASSED ✓"
            else:
                reward = -0.2
                msg = f"Diagnosis submitted. Grader score: {score:.2f} — fix was incomplete."

        # ── Wrong fix actions ────────────────────────────────────────────────

        elif at in ["fix_normalization", "fix_learning_rate",
                    "fix_architecture", "fix_loss_function", "fix_class_balance"]:
            reward = -0.2
            msg = (
                f"'{at}' applied but produced no measurable improvement. "
                f"This may not be the root cause of the issue."
            )

        else:
            reward = -0.1
            msg = f"Unknown action '{at}'. Available actions: {VALID_ACTIONS}"

        return reward, msg

    # ──────────────────────────────────────────────────────────────────────────
    # GRADER
    # ──────────────────────────────────────────────────────────────────────────

    def grade(self):
        # Must have actually fixed labels AND retrained
        if not (self.label_fix_applied and self.retrained_after_fix):
            return 0.0

        y_pred = self.model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred)
        BUGGY_BASELINE = 0.55
        TARGET_ACC     = 0.90
        score = (test_acc - BUGGY_BASELINE) / (TARGET_ACC - BUGGY_BASELINE)
        return _clamp_grade(score)


    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_and_train(self):
        model = LogisticRegression(
            C=1.0,
            max_iter=self.pipeline_state["max_iter"],
            random_state=42
        )
        model.fit(self.X_train, self.y_train_current)
        return model

    def _get_metrics(self):
        """FIX: Use log_loss instead of fake MSE-style loss."""
        train_proba = self.model.predict_proba(self.X_train)
        val_proba   = self.model.predict_proba(self.X_test)
        train_pred  = self.model.predict(self.X_train)
        val_pred    = self.model.predict(self.X_test)
        return {
            "train_acc":  round(accuracy_score(self.y_train_current, train_pred), 4),
            "val_acc":    round(accuracy_score(self.y_test, val_pred), 4),
            "train_loss": round(log_loss(self.y_train_current, train_proba), 4),
            "val_loss":   round(log_loss(self.y_test, val_proba), 4),
        }

    def _build_observation(self, last_action_result="", hint=None):
        metrics = self._get_metrics()
        return {
            "step": self.step_count,
            "task_id": self.TASK_ID,
            "pipeline_state": self.pipeline_state,
            "data_summary": get_data_summary(self.X_train, self.y_train_current),
            "training_metrics": metrics,
            "confidence_score":   self._get_confidence(),
            "last_action_result": last_action_result,
            "available_actions": VALID_ACTIONS,
            "done": self.done,
            "hint": hint if hint is not None else (
                "Try inspecting the data — label quality may be the issue."
                if not self.label_fix_applied else
                "Labels fixed — retrain to see the effect."
            )
        }

    def _check_reasoning_bonus(self, action):
        """+0.05 if reasoning text mentions the correct bug keywords."""
        if not action.reasoning:
            return 0.0
        keywords = REASONING_KEYWORDS.get(action.action_type, [])
        if any(kw in action.reasoning.lower() for kw in keywords):
            return 0.05
        return 0.0

    # def _stuck_detected(self):
    #     """True if last 3 rewards were all negative — agent is going in circles."""
    #     if len(self.last_3_rewards) < 3:
    #         return False
    #     return all(r < 0 for r in self.last_3_rewards)
    # ADD this method to each task class:
    def _get_confidence(self) -> float:
        try:
            proba = self.model.predict_proba(self.X_test)
            return round(float(proba.max(axis=1).mean()), 4)
        except Exception:
            return 0.0