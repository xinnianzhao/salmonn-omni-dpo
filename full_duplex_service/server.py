"""FastAPI wrapper around SALMONN-Omni full-duplex evaluation.

This service deliberately lives in ``salmonn-omni-dpo`` and does not modify the
upstream ``salmonn-omni-dev`` checkout. The current backend wraps
``evaluate_hf.py`` through :class:`orchestrator.full_duplex_client.FullDuplexEvalClient`.

It exposes a stable API now; a future backend can keep SALMONN resident in
memory while preserving the same HTTP contract.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException

from orchestrator.full_duplex_client import FullDuplexEvalClient
from .in_memory_backend import InMemoryFullDuplexGenerator
from .schemas import GenerateBatchRequest, GenerateBatchResponse, GenerateTurnRequest, GenerateTurnResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dev"
DEFAULT_PYTHON = "/data/leuven/362/vsc36212/miniconda3/envs/salmonn/bin/python"
DEFAULT_CHECKPOINT = "/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k"

app = FastAPI(title="DPO Full-Duplex Service")
_client: FullDuplexEvalClient | None = None
_generator: InMemoryFullDuplexGenerator | None = None
_client_lock = Lock()


def _load_json_env(name: str, default_path: str | None = None) -> dict[str, Any]:
    value = os.environ.get(name)
    if not value:
        if not default_path:
            return {}
        path = Path(default_path)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _get_client() -> FullDuplexEvalClient:
    global _client
    if _client is not None:
        return _client


def _get_generator() -> InMemoryFullDuplexGenerator:
    global _generator
    if _generator is not None:
        return _generator
    with _client_lock:
        if _generator is not None:
            return _generator
        _generator = InMemoryFullDuplexGenerator(
            repo_dir=Path(os.environ.get("FULL_DUPLEX_REPO", DEFAULT_REPO)),
            checkpoint_dir=Path(os.environ.get("FULL_DUPLEX_CHECKPOINT", DEFAULT_CHECKPOINT)),
            model_args=_load_json_env(
                "FULL_DUPLEX_MODEL_ARGS_JSON",
                str(PROJECT_ROOT / "configs/full_duplex_default_model_args.json"),
            ),
            data_args=_load_json_env(
                "FULL_DUPLEX_DATA_ARGS_JSON",
                str(PROJECT_ROOT / "configs/full_duplex_default_data_args.json"),
            ),
            eval_args=_load_json_env(
                "FULL_DUPLEX_EVAL_ARGS_JSON",
                str(PROJECT_ROOT / "configs/full_duplex_default_eval_args.json"),
            ),
            device_id=int(os.environ.get("FULL_DUPLEX_DEVICE_ID", "0")),
        )
        return _generator


def _backend_name() -> str:
    return os.environ.get("FULL_DUPLEX_BACKEND", "in_memory").strip().lower()


def _generate_batch(request_dicts: list[dict[str, Any]], batch_id: str) -> dict[str, dict[str, Any]]:
    backend = _backend_name()
    if backend == "in_memory":
        return _get_generator().generate_batch_turns(request_dicts, batch_id=batch_id)
    if backend == "subprocess":
        return _get_client().generate_batch_turns(requests=request_dicts, batch_id=batch_id)
    raise ValueError(f"unsupported FULL_DUPLEX_BACKEND={backend!r}")
    with _client_lock:
        if _client is not None:
            return _client
        _client = FullDuplexEvalClient(
            repo_dir=Path(os.environ.get("FULL_DUPLEX_REPO", DEFAULT_REPO)),
            python_bin=os.environ.get("FULL_DUPLEX_PYTHON", DEFAULT_PYTHON),
            checkpoint_dir=Path(os.environ.get("FULL_DUPLEX_CHECKPOINT", DEFAULT_CHECKPOINT)),
            work_dir=Path(os.environ.get("FULL_DUPLEX_WORK_DIR", str(PROJECT_ROOT / "artifacts/full_duplex_service"))),
            device_id=int(os.environ.get("FULL_DUPLEX_DEVICE_ID", "0")),
            timeout_sec=int(os.environ.get("FULL_DUPLEX_TIMEOUT_SEC", "3600")),
            model_args=_load_json_env(
                "FULL_DUPLEX_MODEL_ARGS_JSON",
                str(PROJECT_ROOT / "configs/full_duplex_default_model_args.json"),
            ),
            data_args=_load_json_env(
                "FULL_DUPLEX_DATA_ARGS_JSON",
                str(PROJECT_ROOT / "configs/full_duplex_default_data_args.json"),
            ),
            eval_args=_load_json_env(
                "FULL_DUPLEX_EVAL_ARGS_JSON",
                str(PROJECT_ROOT / "configs/full_duplex_default_eval_args.json"),
            ),
        )
        return _client


@app.get("/health")
def health() -> dict[str, Any]:
    backend = _backend_name()
    payload = {
        "status": "ok",
        "backend": backend,
        "model_resident": _generator is not None,
        "repo_dir": os.environ.get("FULL_DUPLEX_REPO", DEFAULT_REPO),
        "checkpoint_dir": os.environ.get("FULL_DUPLEX_CHECKPOINT", DEFAULT_CHECKPOINT),
        "device_id": int(os.environ.get("FULL_DUPLEX_DEVICE_ID", "0")),
    }
    if backend == "subprocess":
        client = _get_client()
        payload["work_dir"] = str(client.work_dir)
    return payload


@app.post("/generate_turn")
def generate_turn(request: GenerateTurnRequest) -> GenerateTurnResponse:
    try:
        request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        run_id = request.run_id or f"{request.session_id}_turn{request.turn_id:02d}"
        result = _generate_batch([request_dict], batch_id=run_id)[run_id]
        return GenerateTurnResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": repr(exc), "traceback": traceback.format_exc()},
        ) from exc


@app.post("/generate_batch")
def generate_batch(request: GenerateBatchRequest) -> GenerateBatchResponse:
    try:
        request_dicts = []
        for item in request.requests:
            if hasattr(item, "model_dump"):
                request_dicts.append(item.model_dump())
            else:
                request_dicts.append(item.dict())
        results = _generate_batch(request_dicts, batch_id=request.batch_id)
        return GenerateBatchResponse(
            results={key: GenerateTurnResponse(**value) for key, value in results.items()}
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": repr(exc), "traceback": traceback.format_exc()},
        ) from exc
