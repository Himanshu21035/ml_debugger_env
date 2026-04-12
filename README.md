---
title: ML Pipeline Debugger
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🧠 ML Pipeline Debugger — OpenEnv Environment

An OpenEnv-compliant Reinforcement Learning (RL) environment where an AI agent debugs broken ML training pipelines.

Built for the **Meta PyTorch × Scaler OpenEnv Hackathon 2026**.

---

## 🔗 Links & Badges

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/himanshu21074/ml-debugger-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/huggingface/openenv-core)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

---

## 🚀 Overview

Modern ML pipelines often fail silently due to:
- Mislabeled data
- Incorrect normalization
- Poor hyperparameters
- Wrong loss functions

This project turns ML debugging into an **RL problem**:
> The agent observes a broken pipeline and must identify + fix the root cause.

---

## 🎯 Why This Matters

- 🔬 Identified as an **open research problem** (DeepFix, March 2026)
- 🧪 Fully deterministic grading (fixed seeds, CPU-only)
- ⚡ Dense reward signals for incremental learning
- 🧠 Enables research in **autonomous ML debugging agents**

---

## 🧩 Tasks

| Task | Difficulty | Bug Injected | Success Criterion |
|------|-----------|-------------|-------------------|
| `easy` | Easy | 30% labels flipped | Accuracy > 85% |
| `medium` | Medium | Imbalance + bad normalization + LR=1.0 | Fix all 3 within 10 steps |
| `hard` | Hard | Train normalized, test raw (distribution shift) | Test accuracy > 80% |
| `loss` | Medium | Ridge regression used instead of LogisticRegression (MSE on classification) | Accuracy > 80% after fix |

---

## 📊 Baseline Agent Performance

Hybrid Rule-based + LLM agent (`inference.py`):

| Task | Score | Expected | Result |
|------|-------|----------|--------|
| Easy | **0.871** | 0.75–0.85 | ✅ Exceeded |
| Medium | **0.762** | 0.45–0.60 | 🚀 +27% |
| Hard | **0.960** | 0.20–0.35 | 🔥 +174% |
| Loss | **0.843** | 0.60–0.75 | ✅ Exceeded |

---

## 🌐 API Endpoints

Runs on **FastAPI (port 7860)**

### ▶️ `POST /reset`
Start a new episode.

```json
{
  "task_id": "easy"
}
```

Response:

```json
{
  "episode_id": "abc12345",
  "step": 0,
  "pipeline_state": {},
  "data_summary": {},
  "training_metrics": {},
  "confidence_score": 0.612,
  "available_actions": [],
  "done": false
}
```

---

### ▶️ `POST /step`

Execute an action.

```json
{
  "action_type": "fix_labels",
  "parameter": null,
  "reasoning": "Label flip detected"
}
```

---

### ▶️ `GET /state`

Get episode state.

```json
{
  "step": 4,
  "total_reward": 0.55,
  "current_grade": 0.45,
  "bugs_injected": 1,
  "bugs_fixed": 1
}
```

---

## 🎮 Action Space

| Action | Description |
|--------|-------------|
| `inspect_data` | Analyze dataset statistics |
| `inspect_metrics` | Check train/val performance |
| `inspect_config` | View hyperparameters |
| `inspect_model` | Check model output range + confidence |
| `fix_labels` | Fix flipped/corrupted labels |
| `fix_normalization` | Apply correct feature scaling |
| `fix_learning_rate` | Adjust learning rate |
| `fix_architecture` | Modify model complexity |
| `fix_loss_function` | Correct loss function (e.g. MSE → CrossEntropy) |
| `fix_class_balance` | Handle class imbalance |
| `retrain` | Retrain model with current config |
| `submit_diagnosis` | Declare root cause and end episode |

---

## 💰 Reward Function

| Event | Reward |
|-------|--------|
| New insight from inspection | +0.10 |
| Correct fix applied | +0.20 |
| Improved retraining | +0.30 |
| Final success (grade > 0.8) | +0.50 |
| Correct reasoning bonus | +0.05 |
| Repeated/no-op action | −0.10 |
| Wrong fix applied | −0.20 |
| Step penalty | −0.05 |

---

## ⚡ Quickstart

### ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Run agent:

```bash
export API_KEY=your_key
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

Windows PowerShell:

```powershell
$env:API_KEY="your_key"
$env:API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

---

### 🐳 Run with Docker

```bash
docker build -t ml-debugger-env .
docker run -p 7860:7860 ml-debugger-env
```

---

### ✅ Validate Submission

```bash
pip install openenv-core
openenv validate http://localhost:7860
```

---

## 📁 Project Structure


```
ml_debugger_env/
├── models.py # Pydantic: Action, Observation, Reward, State
├── environment.py # Core logic: reset(), step(), state()
├── app.py # FastAPI server
├── inference.py # Hybrid LLM + rule-based agent
├── client.py # HTTP client for the environment
├── tasks/
│ ├── task_easy.py # Single label-flip bug
│ ├── task_medium.py # 3 simultaneous bugs
│ ├── task_hard.py # Silent distribution shift (PyTorch)
│ └── task_loss.py # Wrong loss function (Ridge → LogisticRegression)
├── data/
│ └── generators.py # Seeded synthetic dataset generators
├── openenv.yaml # OpenEnv metadata
├── Dockerfile
└── requirements.txt

text
```

---

## ⚙️ Technical Details

- **Datasets**: Synthetic (seed=42), fully deterministic
- **Training**: scikit-learn + PyTorch (CPU only)
- **Max Steps**: 15 per episode
- **Server**: FastAPI + Uvicorn, port 7860
- **Python**: 3.11

---

## 🧠 Research Context

Inspired by **DeepFix (2026)**:

> Automated debugging of ML pipelines is still unsolved.

This environment enables:

- Autonomous debugging agents
- Self-healing ML systems
- RL-based reasoning over pipelines

---

## ✅ Submission Checklist

- [x] Environment logic (4 tasks)
- [x] Tasks: easy / medium / hard / loss
- [x] `inspect_model` action + `confidence_score` signal
- [x] Baseline hybrid agent (LLM + rule-based)
- [x] OpenEnv spec compliance (`models.py` inherits base types)
- [x] Docker support
- [x] OpenEnv validation passing (3/3)

---

## ⭐ Future Work

- Add real-world datasets (UCI, Kaggle)
- Multi-step reasoning benchmarks
- Vision + NLP pipeline debugging
- Multi-agent collaborative debugging

---

## 👨‍💻 Author

**Himanshu Singh Baghel**

---



## 📈 Agent Execution Results

### Easy Task
Steps: 5  
Score: 0.871  
Actions: inspect → inspect → inspect → fix_labels → retrain
### Medium Task
Steps: 7  
Score: 0.762  
Actions: inspect → inspect → inspect → fix_balance → fix_norm → fix_lr → retrain
### Hard Task
Steps: 5  
Score: 0.960  
Actions: inspect → inspect → inspect → fix_norm → retrain 

---

## 🖥️ Execution Logs

### Agent Run
![Agent Output](https://github.com/user-attachments/assets/abf49afb-2631-4197-8fcd-2c7128f1de8e)

### OpenEnv Validation (3/3 Passed)
![Validator](https://github.com/user-attachments/assets/1c5860ce-473d-47a0-86c9-07d746a3b8de)

---

