# tasks/task_loss.py
# Task 4 (Loss): Wrong loss function — Ridge regression on binary classification.
# Model trains without errors but outputs garbage probabilities (values > 1 or < 0).
# Agent must detect via confidence_score + metrics and apply fix_loss_function.

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge as _Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from data.generators import get_data_summary

VALID_ACTIONS = [
    "inspect_data", "inspect_metrics", "inspect_config", "inspect_model",
    "fix_labels", "fix_normalization", "fix_learning_rate",
    "fix_architecture", "fix_loss_function", "fix_class_balance",
    "retrain", "submit_diagnosis"
]

REASONING_KEYWORDS = {
    "fix_loss_function": ["loss", "mse", "regression", "cross", "entropy",
                          "probability", "sigmoid", "output"],
}


class LossTask:

    TASK_ID    = "loss"
    MAX_STEPS  = 15
    BUG_CATEGORY = "bad_training_setup"

    def __init__(self):
        self.reset()

    def reset(self):
        np.random.seed(42)
        n = 800
        X = np.random.randn(n, 6).astype(np.float32)
        # Linearly separable binary labels
        y = (X[:, 0] + X[:, 1] * 0.8 - X[:, 2] * 0.5 > 0).astype(int)

        split = int(n * 0.75)
        self.X_train, self.X_test   = X[:split], X[split:]
        self.y_train, self.y_test   = y[:split], y[split:]

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test  = scaler.transform(self.X_test)
        self._scaler = scaler

        self.pipeline_state = {
            "model_type":    "ridge_regression",   # ← the bug
            "loss_function": "mse",                # ← the bug
            "learning_rate": 0.01,
            "normalization": "standard",
        }

        # Bug: Ridge regression on classification → outputs float values not probs
        self._use_correct_loss = False
        self.model = self._build_and_train()

        self.step_count          = 0
        self.done                = False
        self.loss_fix_applied    = False
        self.retrained_after_fix = False
        self.last_inspected      = set()
        self.actions_taken       = []

        print("[ENV] Task LOSS reset. Bug injected: wrong_loss_function (MSE→Ridge)")
        return self._build_observation(
            last_action_result="Episode started. Model trained but outputs look wrong.",
            hint="Model trains without errors but predictions are unreliable. Check model outputs."
        )

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        if self.done:
            return self._build_observation("Episode already finished."), 0.0, True, {}

        self.step_count += 1
        self.actions_taken.append(action.action_type)
        print(f"[ENV] Step {self.step_count}: action={action.action_type} param={action.parameter}")

        reward, result_msg = self._process_action(action)

        bonus = self._check_reasoning_bonus(action)
        if bonus > 0:
            reward     += bonus
            result_msg += f" [+{bonus} reasoning bonus]"

        if self.step_count >= self.MAX_STEPS:
            self.done   = True
            result_msg += " [Max steps reached.]"

        print(f"[ENV] Step {self.step_count} done. reward={reward:.2f} done={self.done}")

        obs  = self._build_observation(last_action_result=result_msg)
        info = {"bugs_fixed": int(self.loss_fix_applied and self.retrained_after_fix),
                "grade":      self.grade()}
        return obs, reward, self.done, info

    # ─────────────────────────────────────────────────────────────────────────

    def _process_action(self, action):
        at     = action.action_type
        reward = -0.05

        if at not in VALID_ACTIONS:
            return -0.1, f"Invalid action '{at}'."

        # ── Inspect ──────────────────────────────────────────────────────────
        if at == "inspect_data":
            if at not in self.last_inspected:
                self.last_inspected.add(at)
                reward = 0.1
                unique, counts = np.unique(self.y_train, return_counts=True)
                msg = (f"Data: {len(self.y_train)} samples. "
                       f"Class dist: {dict(zip(unique.tolist(), counts.tolist()))}. "
                       f"Labels look correct — investigate model outputs instead.")
            else:
                reward = -0.1; msg = "Already inspected data."

        elif at == "inspect_metrics":
            if at not in self.last_inspected:
                self.last_inspected.add(at)
                reward = 0.1
                m   = self._get_metrics()
                raw = self.model.predict(self.X_test)
                out_min, out_max = float(raw.min()), float(raw.max())
                msg = (f"Metrics: train_acc={m['train_acc']:.3f}, val_acc={m['val_acc']:.3f}. "
                       f"Raw model outputs range [{out_min:.2f}, {out_max:.2f}] — "
                       f"probabilities should be in [0, 1]. Something is wrong with the loss function.")
            else:
                reward = -0.1; msg = "Already inspected metrics."

        elif at == "inspect_config":
            if at not in self.last_inspected:
                self.last_inspected.add(at)
                reward = 0.1
                msg = (f"Config: {self.pipeline_state}. "
                       f"model_type=ridge_regression with loss=mse on a classification task — "
                       f"this is incorrect. Use logistic regression with cross-entropy loss.")
            else:
                reward = -0.1; msg = "Already inspected config."

        elif at == "inspect_model":
            if at not in self.last_inspected:
                self.last_inspected.add(at)
                reward = 0.1
                raw    = self.model.predict(self.X_test)
                msg = (
                    f"Model output stats: "
                    f"min={raw.min():.3f}, max={raw.max():.3f}, "
                    f"mean={raw.mean():.3f}, std={raw.std():.3f}. "
                    f"Values outside [0,1] confirm this is a regression model, "
                    f"not a classifier. Fix the loss function."
                )
            else:
                reward = -0.1; msg = "Already inspected model."

        # ── Fix ───────────────────────────────────────────────────────────────
        elif at == "fix_loss_function":
            if not self.loss_fix_applied:
                self._use_correct_loss = True
                self.loss_fix_applied  = True
                self.pipeline_state["model_type"]    = "logistic_regression"
                self.pipeline_state["loss_function"] = "cross_entropy"
                reward = 0.2
                msg    = ("Loss function fixed: switched from Ridge/MSE to "
                          "LogisticRegression/CrossEntropy. Use 'retrain' to validate.")
            else:
                reward = -0.1; msg = "Loss function already fixed."

        elif at == "retrain":
            self.model = self._build_and_train()
            if self.loss_fix_applied:
                self.retrained_after_fix = True
            m = self._get_metrics()
            if m["val_acc"] > 0.80 and self.loss_fix_applied:
                reward    = 0.5
                self.done = True
                msg = (f"Retrain complete. val_acc={m['val_acc']:.3f}. "
                       f"SUCCESS — model now outputs valid probabilities!")
            elif m["val_acc"] > 0.80 and not self.loss_fix_applied:
                reward = -0.1                                    # ← penalise, don't end episode
                msg = (f"Retrain complete. val_acc={m['val_acc']:.3f}. "
                    f"Accuracy looks OK but loss function is still wrong — "
                    f"check model output range with inspect_model.")
            elif m["val_acc"] > 0.65:
                reward = 0.3
                msg    = f"Retrain. val_acc={m['val_acc']:.3f}. Improvement detected."
            else:
                reward = -0.2
                msg    = f"Retrain. val_acc={m['val_acc']:.3f}. Still poor — root cause unresolved."

        elif at == "submit_diagnosis":
            score     = self.grade()
            self.done = True
            if score >= 0.8:
                reward = 0.5
                msg    = f"Diagnosis submitted. Score: {score:.2f} — TASK PASSED ✓"
            else:
                reward = -0.2
                msg    = f"Diagnosis submitted. Score: {score:.2f} — incomplete fix."

        elif at in ["fix_labels", "fix_normalization", "fix_learning_rate",
                    "fix_architecture", "fix_class_balance"]:
            reward = -0.2
            msg    = (f"'{at}' applied — no improvement. "
                      f"This is not the root cause. Investigate model output range.")
        else:
            reward = -0.1; msg = f"Unknown action '{at}'."

        return reward, msg

    # ─────────────────────────────────────────────────────────────────────────

    def grade(self):
        if not (self.loss_fix_applied and self.retrained_after_fix):
            return 0.0
        y_pred   = self.model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred)
        score    = (test_acc - 0.55) / (0.92 - 0.55)
        return round(min(max(score, 0.0), 1.0), 4)

    # ─────────────────────────────────────────────────────────────────────────

    def _build_and_train(self):
        if self._use_correct_loss:
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            model.fit(self.X_train, self.y_train)
        else:
            # Bug: Ridge regression outputs floats, not class probabilities
            model = _RidgeClassifierWrapper()
            model.fit(self.X_train, self.y_train)
        return model

    def _get_metrics(self):
        train_pred = (self.model.predict(self.X_train) > 0.5).astype(int) \
                    if not self._use_correct_loss else self.model.predict(self.X_train)
        val_pred   = (self.model.predict(self.X_test)  > 0.5).astype(int) \
                    if not self._use_correct_loss else self.model.predict(self.X_test)

        has_proba = hasattr(self.model, 'predict_proba')   # ← ADD THIS CHECK
        if self._use_correct_loss and has_proba:
            from sklearn.metrics import log_loss as _log_loss
            train_loss = round(_log_loss(self.y_train, self.model.predict_proba(self.X_train)), 4)
            val_loss   = round(_log_loss(self.y_test,  self.model.predict_proba(self.X_test)),  4)
        else:
            train_loss = round(float(np.mean((self.model.predict(self.X_train) - self.y_train)**2)), 4)
            val_loss   = round(float(np.mean((self.model.predict(self.X_test)  - self.y_test)**2)),  4)


        return {
            "train_acc":  round(accuracy_score(self.y_train, train_pred), 4),
            "val_acc":    round(accuracy_score(self.y_test,  val_pred),   4),
            "train_loss": train_loss,
            "val_loss":   val_loss,
        }

    def _build_observation(self, last_action_result="", hint=None):
        metrics          = self._get_metrics()
        confidence_score = self._get_confidence()
        
        return {
            "step":               self.step_count,
            "task_id":            self.TASK_ID,
            "pipeline_state":     self.pipeline_state,
            "data_summary":       get_data_summary(self.X_train, self.y_train),
            "training_metrics":   metrics,
            "confidence_score":   confidence_score,
            "last_action_result": last_action_result,
            "available_actions":  VALID_ACTIONS,
            "done":               self.done,
            "hint": hint if hint is not None else (
                        "Loss function fixed — retrain to validate."
                        if self.loss_fix_applied else
                        "Model outputs look wrong. Inspect metrics and model outputs carefully."
                    )
        }

    def _get_confidence(self) -> float:
        try:
            has_proba = hasattr(self.model, 'predict_proba')  # ← CHECK IF MODEL HAS PROBABILITY OUTPUT
            if self._use_correct_loss and has_proba:
                proba = self.model.predict_proba(self.X_test)
                return round(float(proba.max(axis=1).mean()), 4)
            else:
                # Ridge outputs raw floats — confidence is meaningless/garbage
                raw = self.model.predict(self.X_test)
                return round(float(np.clip(raw, 0, 1).mean()), 4)
        except Exception:
            return 0.0

    def _check_reasoning_bonus(self, action):
        if not action.reasoning:
            return 0.0
        keywords = REASONING_KEYWORDS.get(action.action_type, [])
        return 0.05 if any(kw in action.reasoning.lower() for kw in keywords) else 0.0


# ── Ridge wrapper to mimic MSE-on-classification ──────────────────────────────
from sklearn.linear_model import Ridge as _Ridge

class _RidgeClassifierWrapper:
    """Wraps Ridge regression to behave like a classifier API (predict only)."""
    def __init__(self):
        self._model = _Ridge(alpha=1.0)

    def fit(self, X, y):
        self._model.fit(X, y.astype(float))
        return self

    def predict(self, X):
        # Returns raw floats — not 0/1 integers
        return self._model.predict(X)