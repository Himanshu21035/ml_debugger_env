# test_task_easy.py
from tasks.task_easy import EasyTask
from models import Action

task = EasyTask()
obs = task.reset()

print("── Initial State ──")
print(f"Step: {obs['step']}, Task: {obs['task_id']}")
print(f"Metrics: {obs['training_metrics']}")
print(f"Hint: {obs['hint']}")
print(f"Initial grade: {task.grade():.4f}  (should be near 0.0)")

print("\n── Agent inspects data ──")
obs, r, done, info = task.step(Action(action_type="inspect_data"))
print(f"Reward: {r}, Result: {obs['last_action_result']}")

print("\n── Agent applies label fix ──")
obs, r, done, info = task.step(Action(action_type="fix_labels"))
print(f"Reward: {r}, Result: {obs['last_action_result']}")

print("\n── Agent retrains ──")
obs, r, done, info = task.step(Action(action_type="retrain"))
print(f"Reward: {r}, Done: {done}")
print(f"Result: {obs['last_action_result']}")
print(f"Final grade: {task.grade():.4f}  (should be > 0.8)")
