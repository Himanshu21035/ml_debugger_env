# environment.py
# Core environment class that wires all 3 tasks together.
# This is what app.py calls — it is the single entry point for:
#   reset(task_id)  → starts a new episode for the given task
#   step(action)    → delegates to the active task
#   state()         → returns episode metadata
#
# OpenEnv contract:
#   POST /reset  → environment.reset()
#   POST /step   → environment.step()
#   GET  /state  → environment.state()

import time
import uuid
from typing import Optional

from models import Action
from tasks.task_easy   import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard   import HardTask


VALID_TASK_IDS = ["easy", "medium", "hard"]


class MLDebuggerEnvironment:
    """
    Manages the full lifecycle of an RL episode.
    One environment instance is shared across all HTTP requests (singleton in app.py).
    reset() must be called before step() — enforced with a guard.
    """

    def __init__(self):
        self._task        = None        # active task object
        self._task_id     = None        # "easy" | "medium" | "hard"
        self._episode_id  = None        # unique ID per episode
        self._started_at  = None        # unix timestamp
        self._step_count  = 0
        self._total_reward = 0.0
        self._reward_history = []       # (step, reward) pairs — for state()
        self._is_done     = False
        self._initialised = False       # guard: step() before reset()

        print("[ENV] MLDebuggerEnvironment created.")

    # ══════════════════════════════════════════════════════════════════════════
    # RESET
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, task_id: str = None) -> dict:
        """
        Start a new episode.

        Args:
            task_id: "easy" | "medium" | "hard"

        Returns:
            First observation dict from the selected task.

        Raises:
            ValueError if task_id is not recognised.
        """
        task_id = (task_id or "easy").lower().strip()
        if task_id not in VALID_TASK_IDS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Must be one of: {VALID_TASK_IDS}"
            )

        # Instantiate the right task
        if task_id == "easy":
            self._task = EasyTask()
        elif task_id == "medium":
            self._task = MediumTask()
        elif task_id == "hard":
            self._task = HardTask()

        # Episode bookkeeping
        self._task_id       = task_id
        self._episode_id    = str(uuid.uuid4())[:8]
        self._started_at    = time.time()
        self._step_count    = 0
        self._total_reward  = 0.0
        self._reward_history = []
        self._is_done       = False
        self._initialised   = True

        print(
            f"[ENV] Episode {self._episode_id} started. "
            f"task={task_id}"
        )

        obs = self._task.reset()
        return self._wrap_observation(obs)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP
    # ══════════════════════════════════════════════════════════════════════════

    def step(self, action: Action) -> tuple:
        """
        Execute one action in the active episode.

        Args:
            action: Action pydantic model with action_type, parameter, reasoning

        Returns:
            (observation dict, reward float, done bool, info dict)

        Raises:
            RuntimeError if called before reset().
        """
        if not self._initialised:
            raise RuntimeError(
                "step() called before reset(). "
                "Call POST /reset with a task_id first."
            )

        if self._is_done:
            # Episode already over — return terminal state, zero reward
            obs = self._task._build_observation("Episode already finished.")
            return self._wrap_observation(obs), 0.0, True, self._build_info()

        # Delegate to active task
        obs, reward, done, info = self._task.step(action)

        # Bookkeeping
        self._step_count  += 1
        self._total_reward += reward
        self._reward_history.append({
            "step":   self._step_count,
            "action": action.action_type,
            "reward": round(reward, 4),
        })
        self._is_done = done

        print(
            f"[ENV] Episode {self._episode_id} | "
            f"step={self._step_count} reward={reward:.2f} "
            f"cumulative={self._total_reward:.2f} done={done}"
        )

        return self._wrap_observation(obs), round(reward, 4), done, self._build_info(info)

    # ══════════════════════════════════════════════════════════════════════════
    # STATE
    # ══════════════════════════════════════════════════════════════════════════

    def state(self) -> dict:
        """
        Returns episode metadata — used by GET /state.
        Safe to call at any time (before or after reset).
        """
        if not self._initialised:
            return {
                "status":        "idle",
                "message":       "No active episode. Call POST /reset to begin.",
                "valid_task_ids": VALID_TASK_IDS,
            }

        elapsed = round(time.time() - self._started_at, 2) if self._started_at else 0

        return {
            "episode_id":     self._episode_id,
            "task_id":        self._task_id,
            "status":         "done" if self._is_done else "active",
            "step":           self._step_count,
            "max_steps":      self._task.MAX_STEPS if self._task else 0,
            "steps_remaining": max(0, (self._task.MAX_STEPS if self._task else 0) - self._step_count),
            "total_reward":   round(self._total_reward, 4),
            "current_grade":  self._task.grade() if self._task else 0.0,
            "elapsed_seconds": elapsed,
            "reward_history": self._reward_history,
            "valid_task_ids": VALID_TASK_IDS,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _wrap_observation(self, obs: dict) -> dict:
        """
        Injects episode-level metadata into every observation.
        Keeps task observations self-contained while adding context.
        """
        obs["episode_id"]      = self._episode_id
        obs["total_reward"]    = round(self._total_reward, 4)
        obs["steps_remaining"] = (
            max(0, (self._task.MAX_STEPS if self._task else 0) - self._step_count)
            if self._task else 0
        )
        return obs

    def _build_info(self, task_info: Optional[dict] = None) -> dict:
        """
        Merges task-specific info with episode-level info.
        Returned as the 4th element of step() tuple.
        """
        info = {
            "episode_id":    self._episode_id,
            "task_id":       self._task_id,
            "step":          self._step_count,
            "total_reward":  round(self._total_reward, 4),
            "current_grade": self._task.grade() if self._task else 0.0,
        }
        if task_info:
            info.update(task_info)
        return info
