# test_models.py
from models import Action, Observation, Reward, State

# Test Action
a = Action(action_type="fix_learning_rate", parameter="0.001", reasoning="LR too high")
print("Action OK:", a)

# Test Observation
o = Observation(
    step=0,
    task_id="easy",
    pipeline_state={"learning_rate": 0.1},
    data_summary={"n_samples": 1000},
    training_metrics={"val_acc": 0.51},
    last_action_result="Episode started.",
    available_actions=["inspect_data", "inspect_metrics", "inspect_config"],
    done=False,
    hint="Something is wrong with the labels."
)
print("Observation OK:", o)

# Test Reward
r = Reward(value=0.1, reason="Inspect revealed useful signal")
print("Reward OK:", r)

# Test State
s = State(episode_id="ep-001", task_id="easy", step_count=0,
          max_steps=15, bugs_injected=1, bugs_fixed=0, done=False)
print("State OK:", s)

print("\n✅ All models valid!")
