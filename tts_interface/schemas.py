"""Request/response models shared by the TTS client and server.

These define the stable wire contract for the ``/synthesize`` API so the DPO
pipeline never depends on CosyVoice's internal signatures directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Supported CosyVoice2 inference modes:
#   zero_shot     - clone a voice from (prompt_text, prompt_audio); needs both.
#   cross_lingual - clone voice from prompt_audio; text language tagged inline
#                   e.g. "<|en|>..."; no prompt_text needed.
#   instruct2     - voice from prompt_audio, style steered by instruct_text.
TTSMode = Literal["zero_shot", "cross_lingual", "instruct2"]


class TTSRequest(BaseModel):
    """A single synthesis request.

    Args:
        text: Text to synthesize. For ``cross_lingual`` it may carry an inline
            language tag, e.g. ``"<|en|>Hello"``.
        output_dir: Directory on the server host where the ``.wav`` is written.
        mode: Which CosyVoice2 inference path to use (see :data:`TTSMode`).
        request_id: Optional caller-supplied id; also the output filename stem.
            A random hex id is generated if omitted.
        prompt_text: Transcript of ``prompt_audio_path`` (required for ``zero_shot``).
        prompt_audio_path: 16 kHz reference wav for voice cloning (required for
            all current modes).
        instruct_text: Natural-language style instruction (required for ``instruct2``).
        speed: Playback speed multiplier passed through to CosyVoice.
        text_frontend: Whether to run CosyVoice's text normalization frontend.
    """

    text: str
    output_dir: str
    mode: TTSMode = "zero_shot"
    request_id: str | None = None
    prompt_text: str | None = None
    prompt_audio_path: str | None = None
    instruct_text: str | None = None
    speed: float = 1.0
    text_frontend: bool = True


class TTSResponse(BaseModel):
    """Result of a synthesis request.

    Args:
        audio_path: Absolute path of the written ``.wav`` on the server host.
        sample_rate: Output sample rate in Hz (from the loaded CosyVoice2 model).
        duration_sec: Length of the generated audio in seconds.
        mode: Echo of the mode used.
        request_id: Echo of the (possibly generated) request id / filename stem.
        text: Echo of the synthesized text.
        metadata: Free-form extras (timing, resolved model/repo paths, params).
    """

    audio_path: str
    sample_rate: int
    duration_sec: float
    mode: TTSMode
    request_id: str
    text: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
