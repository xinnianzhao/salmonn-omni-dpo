from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from .schemas import GenerateBatchRequest, GenerateBatchResponse, GenerateTurnRequest, GenerateTurnResponse


def _model_to_dict(model) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _turn_response_from_dict(data: dict[str, Any]) -> GenerateTurnResponse:
    if hasattr(GenerateTurnResponse, "model_validate"):
        return GenerateTurnResponse.model_validate(data)
    return GenerateTurnResponse.parse_obj(data)


def _batch_response_from_dict(data: dict[str, Any]) -> GenerateBatchResponse:
    if hasattr(GenerateBatchResponse, "model_validate"):
        return GenerateBatchResponse.model_validate(data)
    return GenerateBatchResponse.parse_obj(data)


class FullDuplexServiceError(RuntimeError):
    pass


@dataclass
class FullDuplexServiceClient:
    base_url: str = "http://127.0.0.1:8031"
    timeout_sec: float = 3600.0

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url.rstrip('/')}/health", timeout=10)
        if response.status_code >= 400:
            raise FullDuplexServiceError(f"health failed with HTTP {response.status_code}: {response.text}")
        return response.json()

    def generate_turn(
        self,
        *,
        session_id: str,
        turn_id: int,
        conv_turns: list[dict[str, Any]],
        run_id: str | None = None,
    ) -> GenerateTurnResponse:
        request = GenerateTurnRequest(
            session_id=session_id,
            turn_id=turn_id,
            conv_turns=conv_turns,
            run_id=run_id,
        )
        response = requests.post(
            f"{self.base_url.rstrip('/')}/generate_turn",
            json=_model_to_dict(request),
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            raise FullDuplexServiceError(f"generate_turn failed with HTTP {response.status_code}: {response.text}")
        return _turn_response_from_dict(response.json())

    def generate_batch(self, *, batch_id: str, requests_: list[dict[str, Any]]) -> GenerateBatchResponse:
        request = GenerateBatchRequest(
            batch_id=batch_id,
            requests=[GenerateTurnRequest(**item) for item in requests_],
        )
        response = requests.post(
            f"{self.base_url.rstrip('/')}/generate_batch",
            json=_model_to_dict(request),
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            raise FullDuplexServiceError(f"generate_batch failed with HTTP {response.status_code}: {response.text}")
        return _batch_response_from_dict(response.json())

