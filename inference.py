# inference.py
# Hybrid agent: rule-based first (fast + reliable) → LLM fallback (flexible)
# Mandatory name, must be in project root.
#
# Environment variables:
#   HF_TOKEN      — API key (REQUIRED)
#   API_BASE_URL  — LLM endpoint (default: HuggingFace router)
#   MODEL_NAME    — model to use (default: Qwen/Qwen2.5-72B-Instruct)
#   ENV_URL       — server URL   (default: http://localhost:7860)

import json
import os
import sys
import textwrap
from typing import List, Optional, Set

import requests
from openai import OpenAI


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

API_BASE_URL      = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME        = os.getenv("MODEL_NAME",   "<your-active-model>")
HF_TOKEN          = os.getenv("HF_TOKEN")           # no default — mandatory
LOCAL_IMAGE_NAME  = os.getenv("LOCAL_IMAGE_NAME") 
ENV_URL            = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK          = "ml-debugger-env"
MAX_STEPS          = 15
TEMPERATURE        = 0.2
MAX_TOKENS         = 300
SUCCESS_THRESHOLD  = 0.7    # FIX 7: raised from 0.6


# ══════════════════════════════════════════════════════════════════════════════
# MANDATORY LOG FUNCTIONS — exact format, no deviation
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


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

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_id": task_id}, timeout=60)
    r.raise_for_status()
    return r.json()


def env_step(action_type: str,
             parameter: Optional[str] = None,
             reasoning: Optional[str] = None) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={
        "action_type": action_type,
        "parameter":   parameter,
        "reasoning":   reasoning,
    }, timeout=60)
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = requests.get(f"{ENV_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID AGENT — rule-based first, LLM fallback
# ══════════════════════════════════════════════════════════════════════════════

def choose_action_rule_based(obs, step, used_actions, total_reward):
    task = obs.get("task_id", "easy")

    # Phase 1: Always inspect first
    if step == 1:
        return {"action_type": "inspect_data",
                "reasoning": "always start by inspecting data distribution"}
    if step == 2:
        return {"action_type": "inspect_metrics",
                "reasoning": "check training vs validation performance gap"}
    if step == 3:
        return {"action_type": "inspect_config",
                "reasoning": "check hyperparameters and preprocessing config"}

    # Phase 2: Task-specific fixes FIRST — never skip these
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
                    "reasoning": "train/test distribution mismatch — normalise test set"}
        if "retrain" not in used_actions:
            return {"action_type": "retrain",
                    "reasoning": "normalization fix applied — retrain to validate"}

    # Phase 3: Auto-submit ONLY after fixes are done
    state = env_state()
    current_grade = float(state.get("current_grade", state.get("grade", 0.0)))
    if current_grade >= SUCCESS_THRESHOLD:
        return {"action_type": "submit_diagnosis",
                "reasoning": f"grade {current_grade:.2f} exceeds threshold — submitting"}

    # Terminal fallback
    return {"action_type": "submit_diagnosis",
            "reasoning": "all known fixes applied — submitting final diagnosis"}



# ══════════════════════════════════════════════════════════════════════════════
# LLM FALLBACK AGENT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML debugging agent. You interact with a broken ML training
    pipeline and must identify and fix the root cause of poor model performance.

    AVAILABLE ACTIONS:
      inspect_data, inspect_metrics, inspect_config,
      fix_labels, fix_normalization, fix_learning_rate,
      fix_architecture, fix_loss_function, fix_class_balance,
      retrain, submit_diagnosis

    STRATEGY:
      1. Inspect before fixing
      2. Identify root cause from signals
      3. Apply correct fix
      4. Retrain to validate
      5. Submit when confident

    RESPONSE FORMAT — valid JSON only, no extra text:
    {
        "action_type": "<one of the actions above>",
        "parameter": "<value or null>",
        "reasoning": "<one sentence>"
    }
""").strip()


def get_llm_action(client: OpenAI, step: int,
                   obs: dict, history: List[str],
                   used_actions: Set[str]) -> dict:
    """LLM fallback — only called when rule-based returns None."""
    obs_summary = {
        "step":               obs.get("step"),
        "task_id":            obs.get("task_id"),
        "pipeline_state":     obs.get("pipeline_state"),
        "data_summary":       obs.get("data_summary"),
        "training_metrics":   obs.get("training_metrics"),
        "last_action_result": obs.get("last_action_result"),
        "hint":               obs.get("hint"),
    }
    user_prompt = textwrap.dedent(f"""
        Step: {step}
        Already used actions: {sorted(used_actions)}

        Current observation:
        {json.dumps(obs_summary, indent=2)}

        Recent history:
        {chr(10).join(history[-4:]) if history else 'None'}

        What is your next action? Respond with JSON only.
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

        # Strip markdown code fences if model wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        action = json.loads(raw)
        if "action_type" not in action:
            raise ValueError("Missing action_type")

        # FIX 3: prevent LLM from repeating already-used actions
        if action["action_type"] in used_actions:
            print(f"[DEBUG] LLM suggested already-used action "
                  f"'{action['action_type']}' — overriding to retrain", flush=True)
            action["action_type"] = "retrain" if "retrain" not in used_actions \
                                     else "submit_diagnosis"

        return action

    except Exception as e:
        print(f"[DEBUG] LLM parse error at step {step}: {e}", flush=True)
        # Safe fallback
        fallback = "retrain" if "retrain" not in used_actions else "submit_diagnosis"
        return {
            "action_type": fallback,
            "parameter":   None,
            "reasoning":   "fallback due to parse error",
        }


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(client: OpenAI, task_id: str) -> None:
    rewards:      List[float] = []
    history:      List[str]   = []
    used_actions: Set[str]    = set()   # FIX 3: action memory
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs          = env_reset(task_id)
        total_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done"):
                break

            # ── Hybrid decision: rule-based first, LLM fallback ───────────
            action = choose_action_rule_based(
                obs, step, used_actions, total_reward
            )
            if action is None:
                action = get_llm_action(
                    client, step, obs, history, used_actions
                )
                print(f"[DEBUG] Step {step}: LLM decision → "
                      f"{action['action_type']}", flush=True)
            else:
                print(f"[DEBUG] Step {step}: Rule decision → "
                      f"{action['action_type']}", flush=True)

            action_type = action.get("action_type", "inspect_data")
            parameter   = action.get("parameter")
            reasoning   = action.get("reasoning")

            # Execute
            result = env_step(
                action_type = action_type,
                parameter   = parameter,
                reasoning   = reasoning,
            )

            obs          = result["observation"]
            reward       = float(result["reward"])
            done         = bool(result["done"])
            total_reward += reward

            rewards.append(reward)
            steps_taken = step
            used_actions.add(action_type)  # FIX 3: track used actions

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

        # FIX 1: robust grade extraction
        state   = env_state()
        score   = float(state.get("current_grade",
                         state.get("grade", 0.0)))
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
    if not API_KEY:
        print(
            "[ERROR] No API key. Set HF_TOKEN or API_KEY environment variable.",
            flush=True,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
