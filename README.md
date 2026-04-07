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

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/himanshu21034/ml-debugger-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/huggingface/openenv-core)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

---

## 🚀 Overview

Modern ML pipelines often fail silently due to:
- Mislabeled data  
- Incorrect normalization  
- Poor hyperparameters  

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

| Task   | Difficulty | Bug Injected | Success Criterion |
|--------|-----------|-------------|-------------------|
| `easy` | Easy      | 30% labels flipped | Accuracy > 85% |
| `medium` | Medium  | Imbalance + bad normalization + LR=1.0 | Fix all within 10 steps |
| `hard` | Hard      | Train normalized, test raw | Test accuracy > 80% |

---

## 📊 Baseline Agent Performance

Hybrid Rule-based + LLM agent (`inference.py`):

| Task   | Score  | Expected | Result |
|--------|--------|----------|--------|
| Easy   | **0.871** | 0.75–0.85 | ✅ Exceeded |
| Medium | **0.762** | 0.45–0.60 | 🚀 +27% |
| Hard   | **0.960** | 0.20–0.35 | 🔥 +174% |

---

## 🌐 API Endpoints

Runs on **FastAPI (port 7860)**

### ▶️ `POST /reset`
Start a new episode.


{
  "task_id": "easy"
}


Response:


{
  "episode_id": "abc12345",
  "step": 0,
  "pipeline_state": {...},
  "data_summary": {...},
  "training_metrics": {...},
  "available_actions": [...],
  "done": false
}


---

### ▶️ `POST /step`

Execute an action.


{
  "action_type": "fix_labels",
  "parameter": null,
  "reasoning": "Label flip detected"
}


---

### ▶️ `GET /state`

Get episode state.


{
  "step": 4,
  "total_reward": 0.55,
  "current_grade": 0.45
}


---

## 🎮 Action Space

| Action            | Description          |
| ----------------- | -------------------- |
| inspect_data      | Analyze dataset      |
| inspect_metrics   | Check performance    |
| inspect_config    | View hyperparameters |
| fix_labels        | Fix label errors     |
| fix_normalization | Apply scaling        |
| fix_learning_rate | Adjust LR            |
| fix_architecture  | Modify model         |
| fix_loss_function | Correct loss         |
| fix_class_balance | Handle imbalance     |
| retrain           | Retrain model        |
| submit_diagnosis  | End episode          |

---

## 💰 Reward Function

| Event                       | Reward         |
| --------------------------- | -------------- |
| New insight from inspection | +0.10          |
| Correct fix                 | +0.20 to +0.25 |
| Improved retraining         | +0.30          |
| Final success               | +0.50          |
| No-op action                | −0.10          |
| Worse performance           | −0.20          |
| Step penalty                | −0.05          |

---

## ⚡ Quickstart

### ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Run agent:

```bash
HF_TOKEN=your_token python inference.py
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token"; python inference.py
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
./validate-submission.sh http://localhost:7860 .
```

---

## 📁 Project Structure

```
ml_debugger_env/
├── models.py
├── environment.py
├── app.py
├── inference.py
├── server/
├── tasks/
├── data/
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
└── requirements.txt
```

---

## ⚙️ Technical Details

* **Datasets**: Synthetic (seed=42)
* **Training**: scikit-learn (CPU only)
* **Max Steps**: 15
* **Server**: FastAPI + Uvicorn
* **Python**: 3.11

---

## 🧠 Research Context

Inspired by **DeepFix (2026)**:

> Automated debugging of ML pipelines is still unsolved.

This environment enables:

* Autonomous debugging agents
* Self-healing ML systems
* RL-based reasoning over pipelines

---

## ✅ Submission Checklist

* [x] Environment logic
* [x] Tasks (easy/medium/hard)
* [x] Baseline agent
* [x] OpenEnv config
* [x] Docker support
* [x] Validation passing (3/3)

---

## ⭐ Future Work

* Add real-world datasets
* Multi-step reasoning benchmarks
* Vision + NLP pipelines
* Multi-agent debugging

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

