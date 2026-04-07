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
MAX_STEPS      = 15   # FIX 1: defined here, not on task classes


class MLDebuggerEnvironment:
    """
    Manages the full lifecycle of an RL episode.
    Singleton in app.py — reset() must be called before step().
    """

    def __init__(self):
        self._task           = None
        self._task_id        = None
        self._episode_id     = None
        self._started_at     = None
        self._step_count     = 0
        self._total_reward   = 0.0
        self._reward_history = []
        self._is_done        = False
        self._initialised    = False
        print("[ENV] MLDebuggerEnvironment created.")

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = None) -> dict:
        task_id = (task_id or "easy").lower().strip()
        if task_id not in VALID_TASK_IDS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Must be one of: {VALID_TASK_IDS}"
            )

        if task_id == "easy":
            self._task = EasyTask()
        elif task_id == "medium":
            self._task = MediumTask()
        elif task_id == "hard":
            self._task = HardTask()

        self._task_id        = task_id
        self._episode_id     = str(uuid.uuid4())[:8]
        self._started_at     = time.time()
        self._step_count     = 0
        self._total_reward   = 0.0
        self._reward_history = []
        self._is_done        = False
        self._initialised    = True

        print(f"[ENV] Episode {self._episode_id} started. task={task_id}")

        obs = self._task.reset()
        return self._wrap_observation(obs)

    # ── STEP ──────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> tuple:
        if not self._initialised:
            raise RuntimeError(
                "step() called before reset(). "
                "Call POST /reset with a task_id first."
            )

        # FIX 2: build fallback obs inline, no _build_observation() call
        if self._is_done:
            obs = self._terminal_obs("Episode already finished.")
            return self._wrap_observation(obs), 0.0, True, self._build_info()

        # Max steps guard
        if self._step_count >= MAX_STEPS:
            self._is_done = True
            obs = self._terminal_obs(
                f"Max steps ({MAX_STEPS}) reached. Episode terminated."
            )
            return self._wrap_observation(obs), 0.0, True, self._build_info()

        obs, reward, done, info = self._task.step(action)

        self._step_count   += 1
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

    # ── STATE ─────────────────────────────────────────────────────────────────

    def state(self) -> dict:
        if not self._initialised:
            return {
                "status":         "idle",
                "message":        "No active episode. Call POST /reset to begin.",
                "valid_task_ids": VALID_TASK_IDS,
            }

        elapsed = round(time.time() - self._started_at, 2)

        return {
            "episode_id":      self._episode_id,
            "task_id":         self._task_id,
            "status":          "done" if self._is_done else "active",
            "step":            self._step_count,
            "max_steps":       MAX_STEPS,                 # FIX 1: local constant
            "steps_remaining": max(0, MAX_STEPS - self._step_count),
            "total_reward":    round(self._total_reward, 4),
            "current_grade":   self._task.grade() if self._task else 0.0,
            "elapsed_seconds": elapsed,
            "reward_history":  self._reward_history,
            "valid_task_ids":  VALID_TASK_IDS,
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _terminal_obs(self, message: str) -> dict:
        """FIX 2: minimal safe obs when episode is over — no task method needed."""
        return {
            "step":               self._step_count,
            "task_id":            self._task_id or "unknown",
            "pipeline_state":     {},
            "data_summary":       {},
            "training_metrics":   {},
            "last_action_result": message,
            "available_actions":  [],
            "done":               True,
            "hint":               None,
        }

    def _wrap_observation(self, obs: dict) -> dict:
        """Injects episode-level metadata into every observation."""
        obs["episode_id"]      = self._episode_id
        obs["total_reward"]    = round(self._total_reward, 4)
        obs["steps_remaining"] = max(0, MAX_STEPS - self._step_count)  # FIX 1
        return obs

    def _build_info(self, task_info: Optional[dict] = None) -> dict:
        """Merges task-specific info with episode-level info."""
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

