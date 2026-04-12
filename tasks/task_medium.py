# tasks/task_medium.py



import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.utils import resample
from data.generators import generate_medium_task_data, get_data_summary


VALID_ACTIONS = [
    "inspect_data", "inspect_metrics", "inspect_config", "inspect_model",
    "fix_labels", "fix_normalization", "fix_learning_rate",
    "fix_architecture", "fix_loss_function", "fix_class_balance",
    "retrain", "submit_diagnosis"
]

REASONING_KEYWORDS = {
    "fix_class_balance":  ["imbalance", "class", "weight", "skew", "minority", "uneven"],
    "fix_normalization":  ["scale", "normaliz", "feature", "magnitude", "large", "unit"],
    "fix_learning_rate":  ["learning rate", "lr", "diverge", "unstable", "high", "converge"],
}

# Add this helper at the top of each task file (or in a shared utils):
def _clamp_grade(score: float) -> float:
    """Validator requires strictly (0, 1) — not 0.0, not 1.0."""
    return round(max(0.001, min(score, 0.999)), 4)
def balance_data(X, y):
    X0, X1 = X[y == 0], X[y == 1]
    if len(X0) > len(X1):
        X1 = resample(X1, replace=True, n_samples=len(X0), random_state=42)
    else:
        X0 = resample(X0, replace=True, n_samples=len(X1), random_state=42)
    return np.vstack([X0, X1]), np.array([0]*len(X0) + [1]*len(X1))

class MediumTask:

    TASK_ID   = "medium"
    MAX_STEPS = 15
    TOTAL_BUGS = 3

    def __init__(self):
        self.reset()

    def reset(self):
        (
            self.X_train_buggy,
            self.X_train_clean,
            self.X_test,
            self.y_train,
            self.y_test,
            self.config_buggy,
            self.config_clean
        ) = generate_medium_task_data()

        # Live state — agent's fixes mutate these
        self._inspected_data = False  # must inspect data before diagnosis counts
        self.X_train_current = self.X_train_buggy.copy()
        self.pipeline_state  = self.config_buggy.copy()

        # FIX: track which bugs are actually fixed
        self.bugs_fixed = {
            "class_balance": False,
            "normalization": False,
            "learning_rate": False,
        }

        # Partial observability — signals locked until inspected
        self.revealed = {
            "data":    False,
            "metrics": False,
            "config":  False,
        }

        # FIX: track prev_val_acc manually (was m.get("prev_val_acc", 0) — broken)
        self.prev_val_acc = 0.0
        self.step_count   = 0
        self.done         = False
        self.actions_taken = []
        self.last_3_val_accs = []
        self.last_inspected=set()
        # Fit a scaler on clean data — used when agent applies fix_normalization
        self._scaler = StandardScaler()
        self._scaler.fit(self.X_train_clean)

        self.model = self._build_and_train()

        print(f"[ENV] Task MEDIUM reset. Bugs injected: {self.TOTAL_BUGS}")
        return self._build_observation("Episode started. This pipeline has multiple issues.")

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

        # Early stopping signal (no progress in last 3 retrains)
        if self._no_improvement():
            result_msg += " [No improvement detected in recent steps — try a different approach.]"

        if self.step_count >= self.MAX_STEPS:
            self.done = True
            result_msg += " [Max steps reached — episode ending.]"

        n_fixed = sum(self.bugs_fixed.values())
        print(f"[ENV] Step {self.step_count} done. reward={reward:.2f} bugs_fixed={n_fixed}/3")

        obs  = self._build_observation(last_action_result=result_msg)
        info = {"bugs_fixed": n_fixed, "total_bugs": self.TOTAL_BUGS}
        return obs, reward, self.done, info

    # ──────────────────────────────────────────────────────────────────────────
    # ACTION PROCESSOR
    # ──────────────────────────────────────────────────────────────────────────

    def _process_action(self, action):
        at = action.action_type
        reward = -0.05  # base step penalty

        # ── Validate ─────────────────────────────────────────────────────────
        if at not in VALID_ACTIONS:
            return -0.1, (
                f"Unknown action '{at}'. "
                f"Available actions: {VALID_ACTIONS}"
            )

        # ── Inspect (unlock hidden signals) ──────────────────────────────────

        if at == "inspect_data":
            if not self.revealed["data"]:
                self.revealed["data"] = True
                self._inspected_data = True
                reward = 0.1
                unique, counts = np.unique(self.y_train, return_counts=True)
                # FIX: toned down — no longer says "WARNING: severe class imbalance"
                msg = (
                    f"Data inspection: {len(self.y_train)} samples, "
                    f"{self.X_train_current.shape[1]} features. "
                    f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}. "
                    f"Class ratio may be worth examining. "
                    f"Feature value ranges vary significantly across columns — "
                    f"some features appear much larger than others."
                )
            else:
                reward = -0.1
                msg = "Already inspected data. No new information."

        elif at == "inspect_metrics":
            if not self.revealed["metrics"]:
                self.revealed["metrics"] = True
                reward = 0.1
                m = self._get_metrics()
                # FIX: toned down — no longer says "could be bad LR or scaling issue"
                msg = (
                    f"Metrics: train_acc={m['train_acc']:.3f}, val_acc={m['val_acc']:.3f}, "
                    f"train_loss={m['train_loss']:.4f}, val_loss={m['val_loss']:.4f}, "
                    f"f1_minority={m['f1']:.3f}. "
                    f"There is a noticeable gap between accuracy and F1 score. "
                    f"Training loss behaviour seems inconsistent."
                )
            else:
                reward = -0.1
                msg = "Already inspected metrics. No new information."

        elif at == "inspect_config":
            if not self.revealed["config"]:
                self.revealed["config"] = True
                reward = 0.1
                # FIX: toned down — no longer says "WARNING: LR is very high"
                msg = (
                    f"Config: {self.pipeline_state}. "
                    f"Optimizer settings are visible. "
                    f"Some values may be outside typical ranges for this type of model."
                )
            else:
                reward = -0.1
                msg = "Already inspected config. No new information."

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
        # ── Fix actions ───────────────────────────────────────────────────────

        elif at == "fix_class_balance":
            if not self.bugs_fixed["class_balance"]:
                self.bugs_fixed["class_balance"] = True
                self.pipeline_state["class_weight"] = "balanced"
                reward = 0.2
                msg = (
                    "Class balance fix applied: model will use class_weight='balanced' "
                    "on next retrain. Use 'retrain' to see the effect."
                )
            else:
                reward = -0.1
                msg = "Class balance already fixed. No change."

        elif at == "fix_normalization":
            if not self.bugs_fixed["normalization"]:
                self.bugs_fixed["normalization"] = True
                # FIX: transform X_train_CLEAN, not X_train_current
                # Scaler was fitted on clean data — applying it to buggy data is wrong
                self.X_train_current = self._scaler.transform(self.X_train_current)
                self.X_test = self._scaler.transform(self.X_test)
                reward = 0.2
                new_mean = round(float(self.X_train_current.mean()), 4)
                new_std  = round(float(self.X_train_current.std()), 4)
                msg = (
                    f"Normalization fix applied: StandardScaler fitted on training data. "
                    f"New feature mean={new_mean}, std={new_std}. "
                    f"Use 'retrain' to see the effect."
                )
            else:
                reward = -0.1
                msg = "Normalization already fixed."

        elif at == "fix_learning_rate":
            try:
                new_lr = float(action.parameter) if action.parameter else 0.01
                new_lr = max(1e-5, min(new_lr, 0.5))   # clamp to [1e-5, 0.5]
            except (ValueError, TypeError):
                new_lr = 0.01

            if not self.bugs_fixed["learning_rate"]:
                self.bugs_fixed["learning_rate"] = True
                self.pipeline_state["learning_rate"] = new_lr
                reward = 0.2
                msg = f"Learning rate fixed: set to {new_lr}. Use 'retrain' to apply."
            else:
                self.pipeline_state["learning_rate"] = new_lr
                reward = -0.05
                msg = f"Learning rate updated to {new_lr} (already marked as fixed)."

        elif at == "retrain":
            # FIX: retrain carries an extra -0.05 cost to discourage spamming
            reward = -0.05
            self.model = self._build_and_train()
            m = self._get_metrics()

            # FIX: compare against self.prev_val_acc (not broken m.get())
            improved = m["val_acc"] > self.prev_val_acc + 0.01
            self.last_3_val_accs.append(m["val_acc"])
            if len(self.last_3_val_accs) > 3:
                self.last_3_val_accs.pop(0)

            n_fixed = sum(self.bugs_fixed.values())

            if m["val_acc"] > 0.75 and n_fixed == self.TOTAL_BUGS:
                reward = 0.5
                self.done = True
                msg = (
                    f"Retrain complete. val_acc={m['val_acc']:.3f}, f1={m['f1']:.3f}. "
                    f"All 3 bugs fixed — TASK PASSED ✓"
                )
            elif improved:
                reward += 0.3
                msg = (
                    f"Retrain complete. val_acc={m['val_acc']:.3f}, f1={m['f1']:.3f}. "
                    f"Improvement detected (+{m['val_acc'] - self.prev_val_acc:.3f}). "
                    f"Bugs fixed: {n_fixed}/{self.TOTAL_BUGS}."
                )
            else:
                reward += -0.1
                msg = (
                    f"Retrain complete. val_acc={m['val_acc']:.3f}, f1={m['f1']:.3f}. "
                    f"No meaningful improvement. Bugs fixed: {n_fixed}/{self.TOTAL_BUGS}."
                )

            self.prev_val_acc = m["val_acc"]   # FIX: update tracker

        elif at == "submit_diagnosis":
            score = self.grade()
            self.done = True
            if score >= 0.8:
                reward = 0.5
                msg = f"Diagnosis submitted. Score: {score:.2f} — TASK PASSED ✓"
            elif score >= 0.4:
                reward = score * 0.4   # partial credit
                msg = f"Diagnosis submitted. Score: {score:.2f} — partial credit."
            else:
                reward = -0.1
                msg = f"Diagnosis submitted. Score: {score:.2f} — insufficient fixes applied."

        elif at in ["fix_labels", "fix_architecture", "fix_loss_function"]:
            reward = -0.2
            msg = f"'{at}' produced no improvement — this does not appear to be a root cause here."

        else:
            reward = -0.1
            msg = f"No effect. Available actions: {VALID_ACTIONS}"

        return reward, msg

    # ──────────────────────────────────────────────────────────────────────────
    # GRADER
    # ──────────────────────────────────────────────────────────────────────────

    def grade(self):
        n_fixed   = sum(self.bugs_fixed.values())
        bug_score = n_fixed / self.TOTAL_BUGS

        # Always evaluate with a fresh model reflecting current fixes
        grading_model = self._build_and_train()
        y_pred = grading_model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        score = (bug_score * 0.6) + (f1 * 0.4)
        if n_fixed < self.TOTAL_BUGS:
            score *= 0.7
        return _clamp_grade(score)


    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_and_train(self):
        """
        LR bug simulated via max_iter + tol (not C mapping).
        High LR (1.0) → very few effective iterations → model underfits.
        Correct LR (0.01) → full convergence.
        class_weight actually works now (LogisticRegression supports it natively).
        """
        cw = self.pipeline_state.get("class_weight", None)
        lr = self.pipeline_state.get("learning_rate", 0.01)

        # High LR simulated as early stopping / non-convergence
        if lr >= 0.5:
            max_iter = 5    # model barely converges — simulates LR divergence
            tol      = 1.0   # very loose tolerance — stops immediately
        else:
            max_iter = 500
            tol      = 1e-4

        model = LogisticRegression(
            C=1.0,              # fixed regularization — LR bug is convergence, not C
            class_weight=cw,
            max_iter=1000,
            tol=tol,
            random_state=42,
            solver="lbfgs"
        )
        if self.bugs_fixed["class_balance"]:
            X_fit, y_fit = balance_data(self.X_train_current, self.y_train)
        else:
            X_fit, y_fit = self.X_train_current, self.y_train
        model.fit(X_fit, y_fit)
        return model


    def _get_metrics(self):
        train_pred  = self.model.predict(self.X_train_current)
        val_pred    = self.model.predict(self.X_test)
        train_proba = self.model.predict_proba(self.X_train_current)
        val_proba   = self.model.predict_proba(self.X_test)
        return {
            "train_acc":  round(accuracy_score(self.y_train, train_pred), 4),
            "val_acc":    round(accuracy_score(self.y_test, val_pred), 4),
            "f1":         round(f1_score(self.y_test, val_pred, zero_division=0), 4),
            "train_loss": round(log_loss(self.y_train, train_proba), 4),
            "val_loss":   round(log_loss(self.y_test, val_proba), 4),
        }

    def _build_observation(self, last_action_result=""):
        n_fixed = sum(self.bugs_fixed.values())

        # Partial observability
        if self.revealed["data"]:
            data_summary = get_data_summary(self.X_train_current, self.y_train)
        else:
            data_summary = {
                "n_samples":  int(self.X_train_current.shape[0]),
                "n_features": int(self.X_train_current.shape[1]),
                "note": "Run inspect_data to see full statistics."
            }

        visible_config = {
            "model_type":   "logistic_regression",
            "batch_size":   self.pipeline_state.get("batch_size", 32),
            "epochs":       self.pipeline_state.get("epochs", 50),
            "class_weight": self.pipeline_state.get("class_weight", None),
        }
        if self.revealed["config"]:
            visible_config["learning_rate"] = self.pipeline_state["learning_rate"]
        else:
            visible_config["learning_rate"] = "??? (run inspect_config to reveal)"

        if self.revealed["metrics"]:
            training_metrics = self._get_metrics()
        else:
            training_metrics = {"note": "Run inspect_metrics to see training metrics."}

        return {
            "step":               self.step_count,
            "task_id":            self.TASK_ID,
            "pipeline_state":     visible_config,
            "data_summary":       data_summary,
            "training_metrics":   training_metrics,
            "confidence_score":   self._get_confidence(),
            "last_action_result": last_action_result,
            "available_actions":  VALID_ACTIONS,
            "done":               self.done,
            "hint":               f"Bugs fixed so far: {n_fixed}/{self.TOTAL_BUGS}",
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

    # ADD this method to each task class:
    def _get_confidence(self) -> float:
        try:
            proba = self.model.predict_proba(self.X_test)
            return round(float(proba.max(axis=1).mean()), 4)
        except Exception:
            return 0.0