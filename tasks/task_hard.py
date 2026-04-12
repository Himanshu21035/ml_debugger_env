# # tasks/task_hard.py


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data.generators import generate_distribution_shift_data

# ── Global seed constant ──────────────────────────────────────────────────────
SEED = 42

def set_deterministic():
    """FIX 1: Full determinism — same results every run on any machine."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_deterministic()

# ── Tiny PyTorch model ────────────────────────────────────────────────────────
class TinyNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),  # Dropout can add noise to training, which may hinder debugging clarity
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_torch_model(X_train, y_train, input_dim, epochs=12, lr=0.01):
    """
    FIX 2: Reduced epochs 30→12 for HF CPU speed.
    FIX 1: Seeded DataLoader generator for determinism.
    """
    set_deterministic()
    model     = TinyNet(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)

    # FIX 1: seeded generator ensures identical shuffle order every run
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)

    model.train()
    final_loss = 1.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)

    return model, final_loss


def evaluate_torch_model(model, X, y):
    """Return (loss, accuracy)."""
    model.eval()
    criterion = nn.BCELoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_t)
        loss  = criterion(preds, y_t).item()
        acc   = ((preds > 0.5).float() == y_t).float().mean().item()
    return loss, acc


# ── Task Hard ─────────────────────────────────────────────────────────────────
class TaskHard:
    """
    Silent Distribution Shift:
      - Train is normalized (mean=0, std=1).
      - Test is raw (mean=5, std=10).
      - Model trains well but fails at inference.
    Agent must: detect shift → fix preprocessing → retrain → validate.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        set_deterministic()
        data = generate_distribution_shift_data()

        self.X_train     = data["X_train"]
        self.y_train     = data["y_train"]
        self.X_test_raw  = data["X_test_raw"]
        self.X_test_norm = data["X_test_normalized"]
        self.y_test      = data["y_test"]
        self.train_mean  = data["train_mean"]
        self.train_std   = data["train_std"]
        self.input_dim   = self.X_train.shape[1]
        self.last_inspected = set()  # track which inspect_* actions were called
        # FIX 4: agent must retrain AFTER fixing preprocessing to get credit
        self._inspected_metrics  = False  # must inspect metrics before diagnosis counts
        self._inspected_data     = False  # must inspect data before diagnosis counts
        self._step_num            = 0
        self.shift_detected      = False
        self.preprocessing_fixed = False
        self.retrained           = False
        self.X_test_current      = self.X_test_raw   # starts as raw (broken)
        self.test_acc_after_fix  = None              # only set after retrain

        # Train baseline
        self.model, train_loss = train_torch_model(
            self.X_train, self.y_train, self.input_dim
        )
        _, self.train_acc = evaluate_torch_model(
            self.model, self.X_train, self.y_train
        )
        _, self.test_acc_raw = evaluate_torch_model(
            self.model, self.X_test_raw, self.y_test
        )

        self.pipeline_state = {
            "normalization": "train_only",
            "model_type":    "TinyNet (PyTorch)",
            "input_dim":     self.input_dim,
            "epochs":        12,
            "learning_rate": 0.01,
        }
        self.training_metrics = {
            "train_loss": round(train_loss, 4),
            "train_acc":  round(self.train_acc, 4),
            "val_loss":   None,
            "val_acc":    None,
            "test_acc":   round(self.test_acc_raw, 4),  # suspiciously low
        }
        self.data_summary = {
            "train_shape":       list(self.X_train.shape),
            "test_shape":        list(self.X_test_raw.shape),
            "train_mean_sample": [round(float(v), 3) for v in self.train_mean[:4]],
            "test_mean_sample":  [round(float(v), 3) for v in self.X_test_raw.mean(axis=0)[:4]],
            "distribution_note": "train and test statistics may differ — investigate",
        }
        return self._obs(step=0, result="Task reset. Model trained. Test accuracy is suspiciously low.")

    # ── Actions ───────────────────────────────────────────────────────────────

    def inspect_data(self, **_):
        self._inspected_data = True
        train_mean = self.X_train.mean(axis=0)[:4]
        test_mean  = self.X_test_raw.mean(axis=0)[:4]
        diff       = float(np.abs(train_mean - test_mean).mean())
        flag       = "⚠ LARGE difference — likely distribution shift!" if diff > 0.5 else "Looks similar."
        return (
            f"Train mean (features 0-3): {[round(float(v),3) for v in train_mean]}. "
            f"Test mean (features 0-3):  {[round(float(v),3) for v in test_mean]}. "
            f"Mean abs diff: {diff:.3f}. {flag}"
        ), +0.1

    def inspect_metrics(self, **_):
        gap = self.train_acc - self.test_acc_raw
        self._inspected_metrics = True
        flag = "⚠ Large gap — preprocessing mismatch?" if gap > 0.2 else "Gap acceptable."
        return (
            f"Train acc: {self.train_acc:.3f} | "
            f"Test acc: {self.test_acc_raw:.3f} | "
            f"Gap: {gap:.3f}. {flag}"
        ), +0.1

    def inspect_config(self, **_):
        return (
            f"Pipeline config: {self.pipeline_state}. "
            f"Note: normalization is currently '{self.pipeline_state['normalization']}' — "
            f"test set may not be normalized."
        ), +0.1

    def submit_diagnosis(self, parameter="", reasoning="", **_):
        text = ((parameter or "") + " " + (reasoning or "")).lower()
        
        # Require BOTH "distribution/shift" AND "normalization/preprocessing"
        # Generic reasoning like "applying fix" won't trigger this
        has_shift = (
                "distribution shift" in text or
                ("train" in text and "test" in text) or
                "shift" in text
            )
        has_fix     = any(kw in text for kw in ["normalization", "preprocessing", "normalize"])
        
        # Also require inspect_data was called first
        if not (self._inspected_data and self._inspected_metrics):
            return "⚠ You must inspect data before submitting diagnosis.", -0.1
        
        if has_shift and has_fix:
            self.shift_detected = True
            return "✓ Correct: distribution shift detected.", +0.3
        
        return "✗ Diagnosis too vague. Identify the specific mismatch.", -0.1

    def fix_normalization(self, **_):
        """Apply train mean/std to test set."""
        self.X_test_current      = (self.X_test_raw - self.train_mean) / (self.train_std + 1e-8)
        self.preprocessing_fixed = True
        self.pipeline_state["normalization"] = "train_and_test"
        return (
            "✓ Test set normalized using train statistics. "
            "Now call retrain to apply changes."
        ), +0.2

    def retrain(self, **_):
        """
        FIX 4: test_acc_after_fix ONLY computed here, after preprocessing is fixed.
        If preprocessing not fixed → no reward, no score improvement.
        """
        if not self.preprocessing_fixed:
            return "⚠ Fix preprocessing before retraining. Apply fix_normalization first.", -0.1

        self.model, train_loss = train_torch_model(
            self.X_train, self.y_train, self.input_dim
        )
        _, self.train_acc = evaluate_torch_model(self.model, self.X_train, self.y_train)
        _, test_acc       = evaluate_torch_model(self.model, self.X_test_current, self.y_test)

        self.test_acc_after_fix = test_acc
        self.retrained          = True

        self.training_metrics.update({
            "train_loss": round(train_loss, 4),
            "train_acc":  round(self.train_acc, 4),
            "test_acc":   round(test_acc, 4),
        })
        reward = +0.3 if test_acc > 0.75 else +0.1
        return (
            f"Model retrained on clean data. "
            f"Train acc: {self.train_acc:.3f} | Test acc: {test_acc:.3f} | Loss: {train_loss:.4f}."
        ), reward

    # ── Grader ────────────────────────────────────────────────────────────────

    def grade(self) -> float:
        score = 0.0
        
        # Detection only counts if agent actually inspected first
        if self.shift_detected and self._inspected_data:
            score += 0.3
        
        if self.preprocessing_fixed and self.retrained:
            score += 0.4
            if self.test_acc_after_fix is not None and self.test_acc_after_fix >= 0.75:
                score += 0.3
        
        return round(score, 3)

    # ── Helper ────────────────────────────────────────────────────────────────

    def _obs(self, step: int, result: str, done:bool=False) -> dict:
        return {
            "step":               step,
            "task_id":            "hard",
            "pipeline_state":     self.pipeline_state.copy(),
            "data_summary":       self.data_summary.copy(),
            "training_metrics":   self.training_metrics.copy(),
            "confidence_score":   self._get_confidence(),   # ← ADD THIS LINE
            "last_action_result": result,
            "available_actions":  [
                "inspect_data", "inspect_metrics", "inspect_config",
                "inspect_model",                             # ← ADD THIS TOO
                "submit_diagnosis", "fix_normalization", "retrain",
            ],
            "done": done,
            "hint": None,
        }
    _WRONG = {"fix_labels","fix_learning_rate","fix_class_balance",
               "fix_architecture","fix_loss_function"}
    _VALID = {"inspect_data","inspect_metrics","inspect_config", "inspect_model", "inspect_model", 
               "submit_diagnosis","fix_normalization","retrain"}

    def step(self, action):
        atype     = getattr(action, "action_type", "")
        parameter = getattr(action, "parameter",   None) or ""
        reasoning = getattr(action, "reasoning",   None) or ""
        self._step_num = getattr(self, "_step_num", 0) + 1
        done = False
        reward= -0.05
        if self._step_num > 15:                           
            return self._obs(self._step_num,              
                "Max steps reached."), 0.0, True, self._info()  
        if atype in self._WRONG:
            result = f"'{atype}' not relevant here. Focus on preprocessing."
            return self._obs(self._step_num, result), -0.2, done, self._info()
        if atype not in self._VALID:
            result = f"Unknown action: '{atype}'. Available: {sorted(self._VALID)}"
            return self._obs(self._step_num, result), -0.1, done, self._info()

        if   atype == "inspect_data":      result, reward = self.inspect_data()
        elif atype == "inspect_metrics":   result, reward = self.inspect_metrics()
        elif atype == "inspect_config":    result, reward = self.inspect_config()
        elif atype == "inspect_model":     result, reward = self.inspect_model()
        elif atype == "fix_normalization": result, reward = self.fix_normalization()
        elif atype == "retrain":           result, reward = self.retrain()
        elif atype == "submit_diagnosis":
            result, reward = self.submit_diagnosis(parameter=parameter, reasoning=reasoning)
            done = True
        return self._obs(self._step_num, result), reward, done, self._info()
    def inspect_model(self, **_):
        if "inspect_model" not in self.last_inspected:
            self.last_inspected.add("inspect_model")
            reward = 0.1
            try:
                import torch
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(self.X_test_current, dtype=torch.float32) 
                    proba    = self.model(X_tensor).numpy().flatten()
                conf      = float(np.maximum(proba, 1 - proba).mean())
                low_conf  = float((np.maximum(proba, 1 - proba) < 0.6).mean())
                # Weight norm from first linear layer
                first_layer = list(self.model.parameters())[0]
                coef_norm = float(torch.norm(first_layer).item())
            except Exception:
                conf, low_conf, coef_norm = 0.0, 0.0, 0.0
            msg = (
                f"Model inspection: avg_confidence={conf:.3f}, "
                f"low_confidence_ratio={low_conf:.3f}, "
                f"weight_norm={coef_norm:.3f}. "
                f"{'Low confidence — model likely seeing shifted distribution at test time.' if conf < 0.65 else 'Confidence looks healthy.'}"
            )
        else:
            reward = -0.1
            msg = "Already inspected model. No new information."
        return msg, reward 
    def _info(self):
        return {"shift_detected": self.shift_detected,
                "fix_applied": self.preprocessing_fixed, "grade": self.grade()}

    def _get_metrics(self):
        return {"train_acc": round(self.train_acc, 4),
                "val_acc": self.training_metrics.get("val_acc") or 0.0,
                "test_acc": round(self.test_acc_after_fix or self.test_acc_raw, 4)}

    def _get_confidence(self) -> float:
        try:
            import torch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(self.X_test_current, dtype=torch.float32) 
                proba    = self.model(X_tensor).numpy().flatten()
            confidence = float(np.maximum(proba, 1 - proba).mean())
            return round(confidence, 4)
        except Exception:
            return 0.0

    @property
    def fix_applied(self): return self.preprocessing_fixed

    @property
    def prev_val_acc(self): return 0.0

# Outside class — alias
HardTask = TaskHard