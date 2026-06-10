"""HTTP client for the DPO TTS service.

A small ``requests``-based wrapper around the ``/synthesize`` and ``/health``
endpoints served by :mod:`tts_interface.server`. Pipeline code uses this client
so it never has to construct raw JSON or know CosyVoice's signatures.

The two ``_*_dict`` helpers below keep the client working under both Pydantic
v1 and v2, since the TTS server and its CosyVoice environment may pin different
versions than the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from .schemas import TTSRequest, TTSResponse


def _model_to_dict(model: TTSRequest) -> dict[str, Any]:
    """Serialize a request model to a dict (Pydantic v2 ``model_dump`` or v1 ``dict``)."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _response_from_dict(data: dict[str, Any]) -> TTSResponse:
    """Parse a response dict into a model (Pydantic v2 ``model_validate`` or v1 ``parse_obj``)."""
    if hasattr(TTSResponse, "model_validate"):
        return TTSResponse.model_validate(data)
    return TTSResponse.parse_obj(data)


class TTSRequestError(RuntimeError):
    """Raised when the TTS service returns a non-2xx response."""


@dataclass
class TTSClient:
    """Client for a running TTS service.

    Attributes:
        base_url: Service root, e.g. ``http://127.0.0.1:8021``.
        timeout_sec: Per-request timeout for synthesis (generation can be slow,
            so this defaults high).
    """

    base_url: str = "http://127.0.0.1:8021"
    timeout_sec: float = 300.0

    def synthesize(
        self,
        text: str,
        output_dir: str,
        *,
        mode: str = "zero_shot",
        request_id: str | None = None,
        prompt_text: str | None = None,
        prompt_audio_path: str | None = None,
        instruct_text: str | None = None,
        speed: float = 1.0,
        text_frontend: bool = True,
    ) -> TTSResponse:
        """Synthesize ``text`` to a wav and return its metadata.

        Note ``output_dir`` is interpreted on the *server* host; the returned
        ``audio_path`` points at the file the server wrote. See
        :class:`~tts_interface.schemas.TTSRequest` for per-mode requirements.

        Raises:
            TTSRequestError: if the service responds with HTTP >= 400.
        """
        request = TTSRequest(
            text=text,
            output_dir=output_dir,
            mode=mode,  # type: ignore[arg-type]
            request_id=request_id,
            prompt_text=prompt_text,
            prompt_audio_path=prompt_audio_path,
            instruct_text=instruct_text,
            speed=speed,
            text_frontend=text_frontend,
        )
        response = requests.post(
            f"{self.base_url.rstrip('/')}/synthesize",
            json=_model_to_dict(request),
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            raise TTSRequestError(
                f"TTS request failed with HTTP {response.status_code}: {response.text}"
            )
        return _response_from_dict(response.json())

    def health(self) -> dict[str, Any]:
        """Return the service health payload (status, model_loaded, model_dir).

        Raises:
            TTSRequestError: if the service responds with HTTP >= 400.
        """
        response = requests.get(f"{self.base_url.rstrip('/')}/health", timeout=10)
        if response.status_code >= 400:
            raise TTSRequestError(
                f"TTS health check failed with HTTP {response.status_code}: {response.text}"
            )
        return response.json()
