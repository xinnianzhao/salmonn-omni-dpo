from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateTurnRequest(BaseModel):
    session_id: str
    turn_id: int
    conv_turns: list[dict[str, Any]]
    run_id: str | None = None


class GenerateBatchRequest(BaseModel):
    batch_id: str
    requests: list[GenerateTurnRequest] = Field(default_factory=list)


class GenerateTurnResponse(BaseModel):
    text: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateBatchResponse(BaseModel):
    results: dict[str, GenerateTurnResponse]

