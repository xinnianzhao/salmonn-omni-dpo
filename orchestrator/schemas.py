"""JSON-serialisable records for raw full-duplex DPO data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AudioRef:
    path: str
    sample_rate: int | None = None
    duration_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextResponse:
    text: str
    audio: AudioRef | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResponse:
    candidate_id: str
    text: str
    decode_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRecord:
    turn_id: int
    user: TextResponse
    assistant_direct: TextResponse
    assistant_candidates: list[CandidateResponse] = field(default_factory=list)


@dataclass
class SessionRecord:
    schema_version: str
    session_id: str
    topic: str
    source: dict[str, Any]
    generation: dict[str, Any]
    turns: list[TurnRecord]
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

