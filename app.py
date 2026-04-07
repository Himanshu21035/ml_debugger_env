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
from contextlib import asynccontextmanager
from typing import Optional, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import MLDebuggerEnvironment
from models import Action


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: NumPy-safe JSON encoder
# Converts all numpy scalar/array types to native Python before serialising.
# Without this, any numpy value in obs/info causes TypeError on first request.
# ══════════════════════════════════════════════════════════════════════════════

def np_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives."""
    if isinstance(obj, dict):
        return {k: np_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [np_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def safe_json(content: Any, status_code: int = 200) -> JSONResponse:
    """JSONResponse wrapper that sanitises numpy types first."""
    return JSONResponse(content=np_safe(content), status_code=status_code)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST BODIES
# ══════════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    """All fields optional — validator sends empty body {}."""
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action_type: str
    parameter:   Optional[str] = None
    reasoning:   Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2: lifespan replaces deprecated @app.on_event("startup")
# Also makes smoke test lightweight — no model training on boot.
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: just log — environment is already created at module level
    print("[APP] ML Debugger Environment server starting...")
    print("[APP] Endpoints: POST /reset  POST /step  GET /state")
    print("[APP] Environment instance ready. Waiting for first /reset call.")
    yield
    # Shutdown (nothing to clean up)
    print("[APP] Server shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# APP + MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "ML Debugger Environment",
    description = (
        "OpenEnv-compliant RL environment. Agent debugs broken ML pipelines. "
        "Tasks: Easy (label flip), Medium (3 bugs), Hard (distribution shift)."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,   # FIX 2: modern pattern
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# Singleton — stateful across all requests
env = MLDebuggerEnvironment()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health check."""
    return safe_json({
        "status":      "running",
        "environment": "ML Debugger",
        "version":     "1.0.0",
        "tasks":       ["easy", "medium", "hard"],
        "endpoints": {
            "reset": "POST /reset  — body: {task_id: easy|medium|hard}",
            "step":  "POST /step   — body: {action_type, parameter, reasoning}",
            "state": "GET  /state  — episode metadata",
        },
    })


@app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Start a new episode.
    Empty body {} is valid — defaults to task_id='easy'.
    """
    task_id = "easy"
    if request and request.task_id:
        task_id = request.task_id

    try:
        obs = env.reset(task_id=task_id)
        return safe_json(obs, 200)   # FIX 1: numpy-safe

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e), "valid_task_ids": ["easy", "medium", "hard"]},
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.post("/step")
def step(request: StepRequest):
    """Execute one action. Requires active episode (call /reset first)."""
    try:
        action = Action(
            action_type = request.action_type,
            parameter   = request.parameter,
            reasoning   = request.reasoning,
        )
        obs, reward, done, info = env.step(action)

        return safe_json({           # FIX 1: numpy-safe
            "observation": obs,
            "reward":      reward,
            "done":        done,
            "info":        info,
        }, 200)

    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e), "hint": "Call POST /reset first."},
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e)})


@app.get("/state")
def state():
    """Episode metadata. Safe to call before reset (returns idle status)."""
    try:
        return safe_json(env.state(), 200)   # FIX 1: numpy-safe
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host    = "0.0.0.0",
        port    = 7860,
        reload  = False,
        workers = 1,   # must be 1 — stateful singleton environment
    )

