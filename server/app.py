"""
AuditRx — FastAPI Server
Exposes step() / reset() / state() over HTTP + WebSocket.
Compatible with OpenEnv spec.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import AuditRxEnvironment
from models import AuditRxAction, AuditRxObservation, AuditRxState


# ─────────────────────────────────────────────
# Session management (one env per WS session)
# ─────────────────────────────────────────────

_sessions: dict[str, AuditRxEnvironment] = {}
MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "64"))

VALID_TASKS = ["classify_deviation", "draft_capa", "negotiate_escalation"]


# ─────────────────────────────────────────────
# HTTP request/response wrappers
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "classify_deviation"
    seed: Optional[int] = None
    session_id: Optional[str] = None   # caller can reuse a session


class StepRequest(BaseModel):
    session_id: str
    action: AuditRxAction


class StepResponse(BaseModel):
    observation: AuditRxObservation
    reward: float
    done: bool
    info: dict[str, Any] = {}


class ResetResponse(BaseModel):
    session_id: str
    observation: AuditRxObservation


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="AuditRx — Clinical Trial Deviation Triage",
    description=(
        "OpenEnv-compliant environment for training and evaluating AI agents "
        "on real-world clinical trial protocol deviation triage tasks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HTTP endpoints
# ─────────────────────────────────────────────

_UI_HTML_PATH = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(content=_UI_HTML_PATH.read_text())


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest):
    if req.task_name not in VALID_TASKS:
        raise HTTPException(400, f"Invalid task. Choose from: {VALID_TASKS}")

    session_id = req.session_id or str(uuid4())

    if len(_sessions) >= MAX_CONCURRENT_ENVS and session_id not in _sessions:
        raise HTTPException(503, "Max concurrent sessions reached.")

    env = AuditRxEnvironment(task_name=req.task_name, seed=req.seed)
    obs = env.reset()
    _sessions[session_id] = env

    return ResetResponse(session_id=session_id, observation=obs)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session '{req.session_id}' not found. Call /reset first.")

    obs, reward, done, info = env.step(req.action)

    if done:
        # Keep session alive for state() query; caller closes explicitly
        pass

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state/{session_id}", response_model=AuditRxState)
async def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return env.state()


@app.delete("/close/{session_id}")
async def close(session_id: str):
    env = _sessions.pop(session_id, None)
    if env is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return {"closed": session_id}


# ─────────────────────────────────────────────
# WebSocket endpoint (OpenEnv standard)
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[AuditRxEnvironment] = None
    session_id = str(uuid4())

    try:
        while True:
            data = await websocket.receive_json()
            method = data.get("method")

            if method == "reset":
                task_name = data.get("task_name", "classify_deviation")
                seed      = data.get("seed", None)
                if task_name not in VALID_TASKS:
                    await websocket.send_json({"error": f"Invalid task: {task_name}"})
                    continue
                env = AuditRxEnvironment(task_name=task_name, seed=seed)
                obs = env.reset()
                _sessions[session_id] = env
                await websocket.send_json({
                    "session_id": session_id,
                    "observation": obs.model_dump(),
                })

            elif method == "step":
                if env is None:
                    await websocket.send_json({"error": "Call reset first."})
                    continue
                action = AuditRxAction(**data.get("action", {}))
                obs, reward, done, info = env.step(action)
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                })

            elif method == "state":
                if env is None:
                    await websocket.send_json({"error": "Call reset first."})
                    continue
                await websocket.send_json(env.state().model_dump())

            elif method == "close":
                _sessions.pop(session_id, None)
                await websocket.send_json({"closed": True})
                break

            else:
                await websocket.send_json({"error": f"Unknown method: {method}"})

    except WebSocketDisconnect:
        _sessions.pop(session_id, None)
    except Exception as e:
        _sessions.pop(session_id, None)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
