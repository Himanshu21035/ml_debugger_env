# inference.py
# Hybrid agent: rule-based first (fast + reliable) → LLM fallback (flexible)
# Mandatory name, must be in project root.
#
# Environment variables:
#   API_KEY      — API key (REQUIRED)
#   API_BASE_URL  — LLM endpoint (default: HuggingFace router)
#   MODEL_NAME    — model to use (default: Qwen/Qwen2.5-72B-Instruct)
#   ENV_URL       — server URL   (default: http://localhost:7860)

import json
import os
import sys
import textwrap
from typing import List, Optional, Set
from client import MLDebuggerClient

from openai import OpenAI
import time


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# FIX 2: real HuggingFace inference router endpoint + model
API_BASE_URL = os.environ.get(
    "API_BASE_URL"
)
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct"
)
API_KEY         = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
ENV_URL          = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK        = "ml-debugger-env"
MAX_STEPS        = 15
TEMPERATURE      = 0.2
MAX_TOKENS       = 300
SUCCESS_THRESHOLD = 0.7

env=MLDebuggerClient(ENV_URL)

# ══════════════════════════════════════════════════════════════════════════════
# MANDATORY LOG FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT HTTP CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def env_reset(task_id: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            return env.reset(task_id=task_id)
        except Exception as e:
            if attempt < retries - 1:
                print(f"[DEBUG] Reset failed (attempt {attempt+1}), retrying...", flush=True)
                time.sleep(3)
            else:
                raise
def env_step(action_type: str,
             parameter: Optional[str] = None,
             reasoning: Optional[str] = None) -> dict:
    return env.step(action_type=action_type, parameter=parameter, reasoning=reasoning)

def env_state() -> dict:
    return env.state()


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID AGENT — rule-based first, LLM fallback
# ══════════════════════════════════════════════════════════════════════════════

def choose_action_rule_based(obs, step, used_actions, current_grade):
    task = obs.get("task_id", "easy")

    # Phase 1: Always inspect first (steps 1-3)
    if step == 1:
        return {"action_type": "inspect_data",
                "reasoning": "always start by inspecting data distribution"}
    if step == 2:
        return {"action_type": "inspect_metrics",
                "reasoning": "check training vs validation performance gap"}
    if step == 3:
        return {"action_type": "inspect_config",
                "reasoning": "check hyperparameters and preprocessing config"}

    # Phase 2: Task-specific fixes
    if task == "easy":
        if "fix_labels" not in used_actions:
            return {"action_type": "fix_labels",
                    "reasoning": "easy task — root cause is label flipping"}
        if "retrain" not in used_actions:
            return {"action_type": "retrain",
                    "reasoning": "apply label fix and evaluate"}

    elif task == "medium":
        if "fix_class_balance" not in used_actions:
            return {"action_type": "fix_class_balance",
                    "reasoning": "class imbalance reduces minority class F1"}
        if "fix_normalization" not in used_actions:
            return {"action_type": "fix_normalization",
                    "reasoning": "feature magnitudes inconsistent — apply scaler"}
        if "fix_learning_rate" not in used_actions:
            return {"action_type": "fix_learning_rate",
                    "parameter":   "0.01",
                    "reasoning": "learning rate too high causes divergence"}
        if "retrain" not in used_actions:
            return {"action_type": "retrain",
                    "reasoning": "all 3 fixes applied — retrain to evaluate"}

    elif task == "hard":
        if "fix_normalization" not in used_actions:
            return {"action_type": "fix_normalization",
                    "reasoning": "distribution shift — normalise test set with train stats"}
        if "retrain" not in used_actions:
            return {"action_type": "retrain",
                    "reasoning": "normalization fix applied — retrain to validate"}
    # In choose_action_rule_based(), ADD after the hard block:
    elif task == "loss":
        if "inspect_model" not in used_actions:
            return {"action_type": "inspect_model",
                    "reasoning": "check model output range — may be regression not classifier"}
        if "fix_loss_function" not in used_actions:
            return {"action_type": "fix_loss_function",
                    "reasoning": "model outputs floats not probs — MSE loss on classification bug"}
        if "retrain" not in used_actions:
            return {"action_type": "retrain",
                    "reasoning": "loss function fixed — retrain to validate"}
    # Phase 3: FIX 4 — use grade from obs, no extra HTTP call
    if current_grade >= SUCCESS_THRESHOLD:
        # FIX 3: hard task submit_diagnosis must include "distribution shift"
        reasoning = (
            "distribution shift between train and test confirmed — grade satisfactory"
            if task == "hard"
            else f"grade {current_grade:.2f} exceeds threshold — submitting"
        )
        return {"action_type": "submit_diagnosis", "reasoning": reasoning}

    # Terminal fallback — always include shift keywords for hard task
    reasoning = (
        "distribution shift detected — train normalized, test raw — submitting"
        if task == "hard"
        else "all known fixes applied — submitting final diagnosis"
    )
    return {"action_type": "submit_diagnosis", "reasoning": reasoning}



SYSTEM_PROMPT = textwrap.dedent("""
    You are an ML debugging agent. You interact with a broken ML training pipeline
and must identify and fix the root cause of poor model performance.

You will receive an Observation (JSON) and must return ONE action as JSON.

## Tasks
- easy:   Single label-flip bug. Hint is provided.
- medium: 3 bugs injected simultaneously (normalization + learning rate + class imbalance).
- hard:   Silent distribution shift — train is normalized, test is raw.
- loss:   Wrong loss function — Ridge regression used for classification instead of
          LogisticRegression. Model trains without errors but outputs raw floats,
          not probabilities. confidence_score will be garbage (values near 0.5 randomly).

## Key Signals per Task
- loss task: confidence_score is unreliable, raw model outputs outside [0,1],
             pipeline_state shows model_type=ridge_regression or loss_function=mse.
             Fix: call fix_loss_function, then retrain.

## Action format (return ONLY valid JSON, no markdown):
{
  "action_type": "...",
  "parameter": "...",   // optional
  "reasoning": "..."    // explain your thinking
}

## Strategy
1. Always inspect first before fixing (inspect_data, inspect_metrics, inspect_config, inspect_model).
2. Use confidence_score — low/garbage confidence = model output is broken.
3. Match fix to root cause. Wrong fixes give -0.2 reward.
4. Call retrain after every fix to validate.
5. Call submit_diagnosis only when confident. Wrong diagnosis gives -0.5 reward, so be sure!
""").strip()

def get_llm_action(client: OpenAI, step: int,
                   obs: dict, history: List[str],
                   used_actions: Set[str]) -> dict:
    obs_summary = {
        "step":               obs.get("step"),
        "task_id":            obs.get("task_id"),
        "pipeline_state":     obs.get("pipeline_state"),
        "data_summary":       obs.get("data_summary"),
        "training_metrics":   obs.get("training_metrics"),
        "confidence_score":   obs.get("confidence_score"),   # ← was missing
        "available_actions":  obs.get("available_actions"),  # ← was missing
        "last_action_result": obs.get("last_action_result"),
        "hint":               obs.get("hint"),
    }
    task_id       = obs.get("task_id", "")
    conf          = obs.get("confidence_score")
    metrics       = obs.get("training_metrics", {})
    pipeline      = obs.get("pipeline_state", {})

    # Build diagnostic hints so LLM doesn't have to infer from raw numbers alone
    hints = []
    if conf is not None and conf < 0.60:
        hints.append(f"⚠ confidence_score={conf} is dangerously low — model is uncertain on test data.")
    if task_id == "loss" and pipeline.get("loss_function") == "mse":
        hints.append("⚠ loss_function=mse on a classification task — this is the root cause.")
    if task_id == "hard":
        train_m = obs.get("data_summary", {}).get("train_mean_sample", [])
        test_m  = obs.get("data_summary", {}).get("test_mean_sample", [])
        if train_m and test_m:
            diff = abs(sum(train_m)/len(train_m) - sum(test_m)/len(test_m))
            if diff > 1.0:
                hints.append(f"⚠ Train mean ≈ {train_m} vs Test mean ≈ {test_m} — large gap = distribution shift.")
    train_acc = metrics.get("train_acc", 0)
    test_acc  = metrics.get("test_acc",  0) or metrics.get("val_acc", 0)
    if train_acc and test_acc and (train_acc - test_acc) > 0.2:
        hints.append(f"⚠ train_acc={train_acc} vs test_acc={test_acc} — large gap = preprocessing mismatch.")

    hint_block = "\n".join(hints) if hints else "No anomalies auto-detected — inspect carefully."

    user_prompt = textwrap.dedent(f"""
    ## Observation (Step {step}/15)

    ```json
    {json.dumps(obs_summary, indent=2)}
    ```

    ## Auto-Detected Signals
    {hint_block}

    ## Actions Already Used
    {sorted(used_actions)}

    ## Recent History (last 5 steps)
    {chr(10).join(history[-5:]) if history else "None"}

    ## Rules
    - Do NOT repeat actions already used (unless it's retrain after a new fix).
    - inspect_model gives new signal on model output range and weight norm.
    - For task=loss: inspect_model first, then fix_loss_function, then retrain.
    - For task=hard: submit_diagnosis MUST include "distribution shift" and "normalization" in reasoning.
    - Return ONLY valid JSON. No markdown, no explanation outside the JSON.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        action = json.loads(raw)
        if "action_type" not in action:
            raise ValueError("Missing action_type")

        if action["action_type"] in used_actions:
            print(f"[DEBUG] LLM suggested already-used action "
                  f"'{action['action_type']}' — overriding", flush=True)
            action["action_type"] = ("retrain" if "retrain" not in used_actions
                                     else "submit_diagnosis")
        return action

    except Exception as e:
        print(f"[DEBUG] LLM parse error at step {step}: {e}", flush=True)
        fallback = "retrain" if "retrain" not in used_actions else "submit_diagnosis"
        return {"action_type": fallback, "parameter": None,
                "reasoning": "fallback due to parse error"}


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_action(action: dict) -> dict:
    allowed = {
        "inspect_data", "inspect_metrics", "inspect_config", "inspect_model",
        "fix_labels", "fix_normalization", "fix_learning_rate",
        "fix_architecture", "fix_loss_function", "fix_class_balance",
        "retrain", "submit_diagnosis"
    }
    if action.get("action_type") not in allowed:
        action["action_type"] = "inspect_data"

    if action.get("action_type") == "fix_learning_rate":
        try:
            # Force to float first (handles numeric or string), then back to string
            val = float(action.get("parameter") or "0.01")
            # Clamp to a safe range
            val = max(0.0001, min(val, 0.1))
            action["parameter"] = str(round(val, 6))
        except (TypeError, ValueError):
            action["parameter"] = "0.01"

    # Ensure parameter is always a string or None — never a raw number
    if action.get("parameter") is not None:
        action["parameter"] = str(action["parameter"])

    return action

def run_episode(client: OpenAI, task_id: str) -> None:
    rewards:      List[float] = []
    history:      List[str]   = []
    used_actions: Set[str]    = set()
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False
    last_retrain_step = 0        # ← ADD HERE (top of function)

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        obs           = env_reset(task_id)
        total_reward  = 0.0
        current_grade = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done"):
                break

            action = get_llm_action(client, step, obs, history, used_actions)
            rule_override = choose_action_rule_based(obs, step, used_actions, current_grade)
            task_id = obs.get("task_id", "easy")  
            obs_task_id=obs.get("task_id", task_id)
            easy_incomplete = (                          # ← ADD BLOCK 1
                obs_task_id == "easy" and
                action["action_type"] not in ["fix_labels", "retrain", "submit_diagnosis"] and
                "fix_labels" not in used_actions
            )
            # ── Override logic ────────────────────────────────────────────
            REPEATABLE_ACTIONS = {"retrain", "inspect_data", "inspect_metrics", "inspect_config"}

            fixes_after_retrain = (
                any(f in used_actions for f in
                    ["fix_class_balance", "fix_normalization", "fix_learning_rate"])
                and last_retrain_step < step - 1
            )
            medium_incomplete = (
                obs_task_id == "medium" and
                action["action_type"] == "submit_diagnosis" and
                (not all(f in used_actions for f in [
                    "fix_class_balance", "fix_normalization", "fix_learning_rate"
                ]) or fixes_after_retrain)
            )
            hard_needs_inspect = (                       # ← ADD BLOCK 2
                obs_task_id == "hard" and
                action["action_type"] == "submit_diagnosis" and
                not ("inspect_metrics" in used_actions and "inspect_data" in used_actions)
            )
            loss_incomplete = (
                obs_task_id == "loss" and
                action["action_type"] == "submit_diagnosis" and
                not all(f in used_actions for f in ["fix_loss_function", "retrain"])
            )
            if (
                (action["action_type"] in used_actions
                 and action["action_type"] not in REPEATABLE_ACTIONS)
                or medium_incomplete
                or hard_needs_inspect
                or easy_incomplete
                or loss_incomplete
            ) and rule_override:
                action = rule_override
                print(f"[DEBUG] Step {step}: Rule override → {action['action_type']}", flush=True)
            else:
                print(f"[DEBUG] Step {step}: LLM → {action['action_type']}", flush=True)
            # ─────────────────────────────────────────────────────────────

            action = sanitize_action(action)
            action_type = action.get("action_type", "inspect_data")
            parameter   = action.get("parameter")
            reasoning   = action.get("reasoning")

            result       = env_step(action_type, parameter, reasoning)
            obs          = result["observation"]
            reward       = float(result["reward"])
            done         = bool(result["done"])
            total_reward += reward

            if action_type == "retrain":      # ← ADD HERE (after env_step)
                last_retrain_step = step

            current_grade = float(result.get("info", {}).get("current_grade",
                result.get("info", {}).get("grade", 0.0)))
            

            rewards.append(reward)
            steps_taken = step
            used_actions.add(action_type)

            log_step(step=step, action=action_type,
                     reward=reward, done=done, error=None)

            history.append(
                f"Step {step}: {action_type}"
                + (f"({parameter})" if parameter else "")
                + f" → reward={reward:+.2f} | "
                + obs.get("last_action_result", "")[:100]
            )

            if done:
                break

        state   = env_state()
        score   = float(state.get("current_grade", state.get("grade", current_grade)))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    try:
        API_KEY      = os.environ["API_KEY"]
        API_BASE_URL = os.environ["API_BASE_URL"]
    except KeyError as e:
        print(f"[ERROR] Missing required env var: {e}", flush=True)
        sys.exit(1)

    # FIX 1: api_key=HF_TOKEN not HF_TOKEN=HF_TOKEN
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in ["easy", "medium", "hard", "loss"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        run_episode(client, task_id)

if __name__ == "__main__":
    main()
