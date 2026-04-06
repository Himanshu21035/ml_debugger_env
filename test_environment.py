# test_environment.py
# Tests the environment orchestrator — NOT individual tasks.
# Focuses on: lifecycle, task switching, step guard, state(), wrapping.

from environment import MLDebuggerEnvironment
from models import Action

PASS = "✅"
FAIL = "❌"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f"  →  {detail}" if detail else ""))
    return condition


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ INIT + GUARD ══")
# ══════════════════════════════════════════════════════════════════════════════
env = MLDebuggerEnvironment()

# step() before reset() must raise
try:
    env.step(Action(action_type="inspect_data"))
    check("step() before reset() raises RuntimeError", False)
except RuntimeError as e:
    check("step() before reset() raises RuntimeError", True, str(e)[:50])

# state() before reset() returns idle message
s = env.state()
check("state() before reset() returns idle", s["status"] == "idle")
check("idle state lists valid_task_ids", "valid_task_ids" in s)

# invalid task_id raises
try:
    env.reset(task_id="super_hard")
    check("invalid task_id raises ValueError", False)
except ValueError as e:
    check("invalid task_id raises ValueError", True, str(e)[:60])


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ EASY TASK LIFECYCLE ══")
# ══════════════════════════════════════════════════════════════════════════════
env2 = MLDebuggerEnvironment()
obs  = env2.reset(task_id="easy")

check("reset returns observation dict",  isinstance(obs, dict))
check("obs has episode_id",              "episode_id" in obs)
check("obs has steps_remaining",         "steps_remaining" in obs)
check("obs has total_reward = 0",        obs["total_reward"] == 0.0)
check("obs task_id == easy",             obs["task_id"] == "easy")
check("steps_remaining == MAX_STEPS",    obs["steps_remaining"] == 15)

# state() after reset
s = env2.state()
check("state status == active",          s["status"] == "active")
check("state episode_id matches obs",    s["episode_id"] == obs["episode_id"])
check("state step == 0",                 s["step"] == 0)
check("state total_reward == 0",         s["total_reward"] == 0.0)
check("state current_grade present",     "current_grade" in s)
check("state reward_history == []",      s["reward_history"] == [])

# take one step
obs2, r, done, info = env2.step(Action(
    action_type="inspect_data",
    reasoning="checking data quality"
))
check("step returns 4-tuple",            all([obs2, r is not None, done is not None, info]))
check("obs2 has episode_id",             "episode_id" in obs2)
check("obs2 total_reward updated",       obs2["total_reward"] == round(r, 4))
check("obs2 steps_remaining decreased",  obs2["steps_remaining"] == 14)
check("info has current_grade",          "current_grade" in info)
check("info has episode_id",             "episode_id" in info)

# state() after step
s2 = env2.state()
check("state step == 1",                 s2["step"] == 1)
check("reward_history has 1 entry",      len(s2["reward_history"]) == 1)
check("reward_history entry has action", s2["reward_history"][0]["action"] == "inspect_data")
check("state total_reward == r",         s2["total_reward"] == round(r, 4))


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ TASK SWITCHING ══")
# ══════════════════════════════════════════════════════════════════════════════
env3 = MLDebuggerEnvironment()

# Start easy
env3.reset(task_id="easy")
old_episode = env3.state()["episode_id"]

# Switch to medium mid-episode
obs_m = env3.reset(task_id="medium")
new_episode = env3.state()["episode_id"]

check("reset() on active episode starts fresh",  obs_m["task_id"] == "medium")
check("new episode_id generated",                old_episode != new_episode)
check("step count resets to 0",                  env3.state()["step"] == 0)
check("total_reward resets to 0",                env3.state()["total_reward"] == 0.0)
check("reward_history cleared",                  env3.state()["reward_history"] == [])

# Switch to hard
obs_h = env3.reset(task_id="hard")
check("switch to hard works",                    obs_h["task_id"] == "hard")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ DONE STATE + POST-DONE GUARD ══")
# ══════════════════════════════════════════════════════════════════════════════
env4 = MLDebuggerEnvironment()
env4.reset(task_id="easy")

# Force done via submit_diagnosis
for _ in range(14):   # stay under max_steps
    obs_x, r_x, done_x, _ = env4.step(Action(action_type="submit_diagnosis"))
    if done_x:
        break

check("episode ends after submit_diagnosis", done_x)

# step() after done — should return zero reward, done=True, no crash
obs_post, r_post, done_post, _ = env4.step(Action(action_type="inspect_data"))
check("step after done returns done=True",    done_post)
check("step after done returns reward=0",     r_post == 0.0)

# state() after done
s_done = env4.state()
check("state status == done",                s_done["status"] == "done")
check("state total_reward accumulated",      s_done["total_reward"] != 0.0)


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ FULL EPISODE CUMULATIVE REWARD ══")
# ══════════════════════════════════════════════════════════════════════════════
env5 = MLDebuggerEnvironment()
env5.reset(task_id="medium")

steps_taken = [
    Action(action_type="inspect_data",     reasoning="checking for class imbalance"),
    Action(action_type="inspect_metrics",  reasoning="checking loss divergence"),
    Action(action_type="inspect_config",   reasoning="checking learning rate"),
    Action(action_type="fix_class_balance",reasoning="class imbalance is causing poor minority recall"),
    Action(action_type="fix_normalization",reasoning="feature magnitude is too large, need to normalize scale"),
    Action(action_type="fix_learning_rate",parameter="0.01",
           reasoning="learning rate is too high, causes unstable convergence"),
    Action(action_type="retrain"),
    Action(action_type="submit_diagnosis"),
]

all_rewards = []
for a in steps_taken:
    _, r, done, info = env5.step(a)
    all_rewards.append(r)
    if done:
        break

final_state = env5.state()

check("reward_history length matches steps",
      len(final_state["reward_history"]) == len(all_rewards))

check("state total_reward == sum of step rewards",
      abs(final_state["total_reward"] - round(sum(all_rewards), 4)) < 0.01,
      f"state={final_state['total_reward']:.4f} sum={sum(all_rewards):.4f}")

check("final current_grade > 0.6",
      final_state["current_grade"] > 0.6,
      f"grade={final_state['current_grade']:.4f}")

check("elapsed_seconds > 0",  final_state["elapsed_seconds"] > 0)

print(f"\n  Episode summary:")
print(f"    episode_id:   {final_state['episode_id']}")
print(f"    steps taken:  {final_state['step']}")
print(f"    total_reward: {final_state['total_reward']}")
print(f"    final_grade:  {final_state['current_grade']}")
print(f"    elapsed:      {final_state['elapsed_seconds']}s")
