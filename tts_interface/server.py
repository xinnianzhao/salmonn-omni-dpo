"""FastAPI service wrapping CosyVoice2 for the DPO pipeline.

---------------
CosyVoice2 lives in the external SALMONN-Omni dev repo and pulls in heavy,
version-pinned dependencies (and a multi-GB checkpoint). Rather than vendor any
of that here, this service loads the model **once** and exposes a small, stable
HTTP API (``/health`` + ``/synthesize``). The DPO pipeline only ever talks to
that API via :class:`tts_interface.client.TTSClient`.

--------------------------
The CosyVoice package is not importable by default; it sits inside the dev repo
under ``models/``. ``_prepare_imports`` injects the right directories onto
``sys.path`` at first use. Paths are configurable via environment variables:

    COSYVOICE_REPO_DIR    root of the SALMONN-Omni dev repo (contains models/)
    COSYVOICE_MODEL_DIR   the CosyVoice2-0.5B checkpoint directory
    COSYVOICE_LOAD_JIT    "1" to load the JIT-compiled model
    COSYVOICE_LOAD_TRT    "1" to load the TensorRT engine
    COSYVOICE_FP16        "1" to run the model in fp16

Run with::

    PYTHONPATH=$PWD bash scripts/start_tts_service.sh
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import types
import uuid
from pathlib import Path
from threading import Lock

import torch
import torchaudio
from fastapi import FastAPI, HTTPException

from .schemas import TTSRequest, TTSResponse


# Default locations on the VSC scratch/project filesystem. Both are overridable
# via the COSYVOICE_* environment variables documented in the module docstring.
DEFAULT_COSYVOICE_REPO = (
    "/lustre1/scratch/362/vsc36212/projects/salmonn-omni/"
    "salmonn-omni-dev-mine-demo"
)
DEFAULT_MODEL_DIR = (
    "/lustre1/project/stg_00197/projects/salmonn-omni/"
    "playground/tts/CosyVoice2-0.5B"
)

app = FastAPI(title="DPO TTS Service")

# Lazily-loaded CosyVoice2 singleton. Loading is deferred until the first
# request (so the process can start and answer /health before the model is
# resident) and guarded by a lock so concurrent requests load it only once.
_model = None
_model_lock = Lock()


def _prepare_imports() -> None:
    """Put the CosyVoice packages on ``sys.path``.

    The package lives at ``<repo>/models/cosyvoice`` and vendors Matcha-TTS
    under ``third_party``; both directories must be importable.
    """
    repo_dir = os.environ.get("COSYVOICE_REPO_DIR", DEFAULT_COSYVOICE_REPO)
    models_dir = os.path.join(repo_dir, "models")
    matcha_dir = os.path.join(repo_dir, "third_party", "Matcha-TTS")
    for path in (models_dir, repo_dir, matcha_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_model():
    """Load (or return the cached) CosyVoice2 model, thread-safely."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        # Double-checked: another thread may have loaded it while we waited.
        if _model is not None:
            return _model
        _prepare_imports()

        # Some CosyVoice builds reference ``is_only_punctuation`` from the
        # frontend module without importing it there. Backfill the symbol so
        # the import doesn't blow up at runtime.
        import cosyvoice.cli.frontend as cosy_frontend
        from cosyvoice.utils.frontend_utils import is_only_punctuation

        if not hasattr(cosy_frontend, "is_only_punctuation"):
            cosy_frontend.is_only_punctuation = is_only_punctuation

        from cosyvoice.cli.cosyvoice import CosyVoice2

        model_dir = os.environ.get("COSYVOICE_MODEL_DIR", DEFAULT_MODEL_DIR)
        load_jit = os.environ.get("COSYVOICE_LOAD_JIT", "0") == "1"
        load_trt = os.environ.get("COSYVOICE_LOAD_TRT", "0") == "1"
        fp16 = os.environ.get("COSYVOICE_FP16", "0") == "1"
        _model = CosyVoice2(model_dir, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
        _patch_cosyvoice2_model(_model)
        return _model


def _patch_cosyvoice2_model(cosyvoice) -> None:
    """Normalize ``token2wav`` to always return just the speech tensor.

    Depending on the CosyVoice build, ``model.token2wav`` may return either a
    bare tensor or a ``(speech, ...)`` tuple. We wrap it once to unwrap the
    tuple form so the synthesis path downstream only deals with tensors. The
    ``_dpo_token2wav_patched`` flag makes this idempotent.
    """
    model = getattr(cosyvoice, "model", None)
    if model is None or getattr(model, "_dpo_token2wav_patched", False):
        return

    original_token2wav = model.token2wav

    def token2wav_tensor(self, *args, **kwargs):
        result = original_token2wav(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    model.token2wav = types.MethodType(token2wav_tensor, model)
    model._dpo_token2wav_patched = True


def _load_prompt_wav(path: str):
    """Load a reference wav resampled to 16 kHz (CosyVoice's expected rate)."""
    _prepare_imports()
    from cosyvoice.utils.file_utils import load_wav

    return load_wav(path, 16000)


def _validate_request(request: TTSRequest) -> None:
    """Enforce per-mode required fields, raising HTTP 400 on violations."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if request.mode in {"zero_shot", "cross_lingual", "instruct2"} and not request.prompt_audio_path:
        raise HTTPException(status_code=400, detail=f"prompt_audio_path is required for {request.mode}")
    if request.mode == "zero_shot" and not request.prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required for zero_shot")
    if request.mode == "instruct2" and not request.instruct_text:
        raise HTTPException(status_code=400, detail="instruct_text is required for instruct2")


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report liveness and whether the model has finished loading."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_dir": os.environ.get("COSYVOICE_MODEL_DIR", DEFAULT_MODEL_DIR),
    }


@app.post("/synthesize")
def synthesize(request: TTSRequest) -> TTSResponse:
    """Synthesize speech for one request and write it to ``output_dir``.

    Dispatches to the CosyVoice2 inference method matching ``request.mode``,
    concatenates the (non-streaming) output chunks into one waveform, saves it
    as ``<output_dir>/<request_id>.wav``, and returns its metadata.
    """
    _validate_request(request)
    cosyvoice = _load_model()

    request_id = request.request_id or uuid.uuid4().hex
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{request_id}.wav"

    prompt_speech_16k = _load_prompt_wav(request.prompt_audio_path) if request.prompt_audio_path else None
    start = time.time()

    try:
        # All modes run non-streaming (stream=False): we want the complete
        # waveform in one shot for offline DPO data generation, not low-latency
        # playback. Each inference_* call yields dicts with a "tts_speech" tensor.
        if request.mode == "zero_shot":
            generator = cosyvoice.inference_zero_shot(
                request.text,
                request.prompt_text or "",
                prompt_speech_16k,
                stream=False,
                speed=request.speed,
                text_frontend=request.text_frontend,
            )
        elif request.mode == "cross_lingual":
            generator = cosyvoice.inference_cross_lingual(
                request.text,
                prompt_speech_16k,
                stream=False,
                speed=request.speed,
                text_frontend=request.text_frontend,
            )
        elif request.mode == "instruct2":
            generator = cosyvoice.inference_instruct2(
                request.text,
                request.instruct_text or "",
                prompt_speech_16k,
                stream=False,
                speed=request.speed,
                text_frontend=request.text_frontend,
            )
        else:
            raise HTTPException(status_code=400, detail=f"unsupported mode: {request.mode}")

        # Concatenate all chunks along the time axis into a single waveform.
        chunks = [item["tts_speech"] for item in generator if item]
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio chunks")
        speech = torch.cat(chunks, dim=1)
        torchaudio.save(str(audio_path), speech.cpu(), sample_rate=cosyvoice.sample_rate)
    except HTTPException:
        # Validation/dispatch errors already carry the right status code.
        raise
    except Exception as exc:
        # Surface unexpected failures as HTTP 500 with the traceback in logs.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    duration_sec = speech.shape[1] / float(cosyvoice.sample_rate)
    return TTSResponse(
        audio_path=str(audio_path),
        sample_rate=int(cosyvoice.sample_rate),
        duration_sec=duration_sec,
        mode=request.mode,
        request_id=request_id,
        text=request.text,
        metadata={
            "stream": False,
            "speed": request.speed,
            "text_frontend": request.text_frontend,
            "elapsed_sec": round(time.time() - start, 3),
            "cosyvoice_repo_dir": os.environ.get("COSYVOICE_REPO_DIR", DEFAULT_COSYVOICE_REPO),
            "cosyvoice_model_dir": os.environ.get("COSYVOICE_MODEL_DIR", DEFAULT_MODEL_DIR),
        },
    )
