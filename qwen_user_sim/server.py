"""FastAPI wrapper exposing the Qwen user simulator as a small HTTP service.

This is an optional convenience layer on top of the raw vLLM endpoint: instead
of each pipeline component re-implementing the prompt construction and OpenAI
call, they POST the dialogue state to ``/generate_user`` and get back a single
simulated user turn.

Layout (default ports)::

    vLLM OpenAI API   :8001  ->  QwenUserSimulator  ->  this service  :8011

Configuration is via environment variables (read once, see ``_simulator``):

    QWEN_BASE_URL     vLLM OpenAI-compatible base URL  (default :8001/v1)
    QWEN_MODEL        served model name                (default qwen3-32b-user-sim)
    QWEN_TEMPERATURE  sampling temperature             (default 0.6)
    QWEN_TOP_P        nucleus-sampling cutoff          (default 0.9)
    QWEN_MAX_TOKENS   max utterance length             (default 128)

Run with::

    bash scripts/start_qwen_user_sim_api.sh
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .client import QwenUserSimulator


class HistoryItem(BaseModel):
    """One prior dialogue turn."""

    role: str  # "user" or "assistant"
    text: str


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate_user``."""

    topic: str
    history: list[HistoryItem] = Field(default_factory=list)
    turn_id: int = 1
    user_profile: dict[str, Any] | None = None


app = FastAPI(title="Qwen3 User Simulator")


@lru_cache(maxsize=1)
def _simulator() -> QwenUserSimulator:
    """Build the simulator once and reuse it across requests.

    Caching avoids re-creating the underlying ``OpenAI`` HTTP client on every
    call. Environment variables are therefore read a single time at first use;
    they are fixed for the lifetime of the process anyway.
    """
    return QwenUserSimulator(
        base_url=os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8001/v1"),
        model=os.environ.get("QWEN_MODEL", "qwen3-32b-user-sim"),
        temperature=float(os.environ.get("QWEN_TEMPERATURE", "0.6")),
        top_p=float(os.environ.get("QWEN_TOP_P", "0.9")),
        max_tokens=int(os.environ.get("QWEN_MAX_TOKENS", "128")),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/generate_user")
def generate_user(request: GenerateRequest) -> dict[str, Any]:
    """Generate the next simulated user utterance for the given dialogue state."""
    return _simulator().generate_user_turn(
        topic=request.topic,
        history=[item.model_dump() for item in request.history],
        turn_id=request.turn_id,
        user_profile=request.user_profile,
    )
