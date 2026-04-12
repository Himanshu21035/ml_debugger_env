# models.py
# Pydantic models define the "shape" of all data exchanged between
# the agent (client) and the environment (server).
# Inherits from openenv-core base types for spec compliance.

from typing import Optional, List
from pydantic import BaseModel

try:
    from openenv.core import Action      as _Action
    from openenv.core import Observation as _Observation
    from openenv.core import State       as _State
except ImportError:
    # Fallback if openenv-core not installed locally
    _Action      = BaseModel
    _Observation = BaseModel
    _State       = BaseModel


# ── ACTION ────────────────────────────────────────────────────────────────────
# What the agent sends to the environment each step.

class Action(_Action):
    action_type: str
    # Must be one of these strings:
    # "inspect_data"       → look at dataset stats
    # "inspect_metrics"    → look at train/val loss and accuracy
    # "inspect_config"     → look at current hyperparameters/architecture
    # "fix_labels"         → relabel corrupted/flipped labels
    # "fix_normalization"  → apply correct feature scaling
    # "fix_learning_rate"  → change the learning rate
    # "fix_architecture"   → change model complexity
    # "fix_loss_function"  → swap to correct loss
    # "fix_class_balance"  → apply class weights or resampling
    # "retrain"            → retrain the model with current config
    # "submit_diagnosis"   → declare the bug(s) found and end episode

    parameter: Optional[str] = None
    # Extra info for the action, e.g.:
    # fix_learning_rate → parameter="0.001"
    # fix_normalization → parameter="standard"

    reasoning: Optional[str] = None
    # The agent's explanation (logged but not graded)

    class Config:
        extra = "allow"


# ── OBSERVATION ───────────────────────────────────────────────────────────────
# What the environment returns to the agent after each reset() or step().

class Observation(_Observation):
    step: int                          # Current step number (0 at reset)
    task_id: str                       # Which task: "easy", "medium", "hard", "loss"
    pipeline_state: dict               # e.g. {"learning_rate": 0.1, "batch_size": 32, ...}
    data_summary: dict                 # e.g. {"n_samples": 1000, "class_dist": {...}, ...}
    training_metrics: dict             # e.g. {"train_loss": 0.9, "val_acc": 0.51, ...}
    last_action_result: str            # Human-readable result of the last action taken
    available_actions: List[str]       # Which action_types are valid right now
    done: bool                         # True = episode is over
    hint: Optional[str] = None         # Only shown on Task 1 (Easy)
    confidence_score: Optional[float] = None  # Mean max predicted probability

    class Config:
        extra = "allow"


# ── REWARD ────────────────────────────────────────────────────────────────────
# The reward signal returned alongside each Observation.

class Reward(BaseModel):              # Reward has no openenv base type
    value: float                      # The numeric reward (can be negative)
    reason: str                       # Why this reward was given (for logging/debugging)


# ── STATE ─────────────────────────────────────────────────────────────────────
# Episode metadata returned by GET /state.

class State(_State):
    episode_id: str        # Unique ID for this episode
    task_id: str           # Which task is active
    step_count: int        # How many steps have been taken
    max_steps: int         # Episode ends forcibly after this many steps (15)
    bugs_injected: int     # Total bugs in this episode
    bugs_fixed: int        # How many the agent has correctly fixed so far
    done: bool             # Whether the episode has ended
    current_grade: float = 0.0   # Latest grader score 0.0–1.0
    total_reward: float  = 0.0   # Cumulative reward this episode

    class Config:
        extra = "allow"