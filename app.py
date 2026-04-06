# app.py
# FastAPI server — the actual submission artifact.
# Runs on port 7860 (HuggingFace Spaces default).
#
# Three endpoints — this is the full OpenEnv contract:
#   POST /reset  → start new episode, returns first observation
#   POST /step   → take one action, returns (obs, reward, done, info)
#   GET  /state  → episode metadata
#
# CRITICAL NOTES from official validator:
#   - POST /reset with empty body {} must return HTTP 200 (validator Step 1)
#   - All responses must be JSON-serialisable
#   - Single environment instance shared across requests (stateful server)

from __future__ import annotations

import traceback
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import MLDebuggerEnvironment
from models import Action


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST BODIES
# ══════════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    """
    All fields optional — validator sends empty body {}.
    task_id defaults to "easy" so POST /reset always works.
    """
    task_id: Optional[str] = "easy"


class StepRequest(BaseModel):
    """
    Maps directly to our Action model.
    Kept separate so FastAPI generates clean API docs.
    """
    action_type: str
    parameter:   Optional[str] = None
    reasoning:   Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# APP + MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "ML Debugger Environment",
    description = (
        "An OpenEnv-compliant RL environment where an AI agent debugs "
        "broken ML training pipelines. Three tasks: Easy (label flip), "
        "Medium (3 simultaneous bugs), Hard (silent distribution shift)."
    ),
    version     = "1.0.0",
)

# Allow all origins — required for HF Spaces iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Single shared environment instance — stateful across requests
env = MLDebuggerEnvironment()

print("[APP] ML Debugger Environment server initialised.")
print("[APP] Endpoints: POST /reset  POST /step  GET /state")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health check — also shows available tasks."""
    return {
        "status":       "running",
        "environment":  "ML Debugger",
        "version":      "1.0.0",
        "tasks":        ["easy", "medium", "hard"],
        "endpoints":    {
            "reset": "POST /reset  — start episode, body: {task_id: easy|medium|hard}",
            "step":  "POST /step   — take action, body: {action_type, parameter, reasoning}",
            "state": "GET  /state  — episode metadata",
        },
    }


@app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Start a new episode.

    Body (all optional):
        task_id: "easy" | "medium" | "hard"  (default: "easy")

    Returns:
        First observation dict for the selected task.
    """
    # Handle completely empty body (validator sends {})
    task_id = "easy"
    if request and request.task_id:
        task_id = request.task_id

    try:
        obs = env.reset(task_id=task_id)
        return JSONResponse(content=obs, status_code=200)

    except ValueError as e:
        # Unknown task_id — return 400 with helpful message
        raise HTTPException(
            status_code=400,
            detail={
                "error":          str(e),
                "valid_task_ids": ["easy", "medium", "hard"],
            }
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post("/step")
def step(request: StepRequest):
    """
    Execute one action in the active episode.

    Body:
        action_type: str       — required
        parameter:   str|null  — optional (e.g. learning rate value)
        reasoning:   str|null  — optional (used for reasoning bonus)

    Returns:
        {observation, reward, done, info}
    """
    try:
        action = Action(
            action_type = request.action_type,
            parameter   = request.parameter,
            reasoning   = request.reasoning,
        )
        obs, reward, done, info = env.step(action)

        return JSONResponse(
            content={
                "observation": obs,
                "reward":      reward,
                "done":        done,
                "info":        info,
            },
            status_code=200,
        )

    except RuntimeError as e:
        # step() before reset()
        raise HTTPException(
            status_code=400,
            detail={
                "error":   str(e),
                "hint":    "Call POST /reset first to start an episode.",
            }
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/state")
def state():
    """
    Get current episode metadata.
    Safe to call at any time — returns idle status before reset().

    Returns:
        Episode metadata including step count, total reward, grade, etc.
    """
    try:
        return JSONResponse(content=env.state(), status_code=200)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP EVENT — smoke test on boot
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup_smoke_test():
    """
    Runs once when server boots.
    Verifies all 3 tasks initialise without error.
    If this fails, the server still starts — just logs the warning.
    """
    print("[APP] Running startup smoke test...")
    try:
        for task_id in ["easy", "medium", "hard"]:
            test_env = MLDebuggerEnvironment()
            obs = test_env.reset(task_id=task_id)
            assert obs["task_id"] == task_id
            assert "available_actions" in obs
            print(f"[APP]   ✓ task={task_id} OK")
        print("[APP] Smoke test passed. Server ready.")
    except Exception as e:
        print(f"[APP] ⚠ Smoke test warning: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host    = "0.0.0.0",
        port    = 7860,       # HuggingFace Spaces default
        reload  = False,      # never True in production
        workers = 1,          # single worker — environment is stateful
    )
