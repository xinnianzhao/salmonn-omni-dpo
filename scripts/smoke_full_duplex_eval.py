#!/usr/bin/env python3
"""Smoke test the SALMONN-Omni full-duplex evaluator only."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.full_duplex_client import FullDuplexEvalClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-duplex-repo", default="../salmonn-omni-dev")
    parser.add_argument("--full-duplex-python", default="python")
    parser.add_argument(
        "--checkpoint-dir",
        default="/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k",
    )
    parser.add_argument("--model-args-json", default=str(PROJECT_ROOT / "configs/full_duplex_default_model_args.json"))
    parser.add_argument("--data-args-json", default=str(PROJECT_ROOT / "configs/full_duplex_default_data_args.json"))
    parser.add_argument("--eval-args-json", default=str(PROJECT_ROOT / "configs/full_duplex_default_eval_args.json"))
    parser.add_argument("--user-audio", default=str(PROJECT_ROOT / "asset/user_prompt.wav"))
    parser.add_argument("--artifact-dir", default="artifacts/full_duplex_only_smoke")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def load_json(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json).resolve() if args.output_json else artifact_dir / "result.json"

    session_id = f"full_duplex_smoke_{uuid.uuid4().hex[:8]}"
    client = FullDuplexEvalClient(
        repo_dir=Path(args.full_duplex_repo),
        python_bin=args.full_duplex_python,
        checkpoint_dir=Path(args.checkpoint_dir),
        work_dir=artifact_dir / "eval_runs",
        device_id=args.device_id,
        model_args=load_json(args.model_args_json),
        data_args=load_json(args.data_args_json),
        eval_args=load_json(args.eval_args_json),
    )
    result = client.generate_turn(
        session_id=session_id,
        turn_id=1,
        conv_turns=[
            {
                "user": "Please briefly introduce the Outer Banks in North Carolina.",
                "user_path": str(Path(args.user_audio).resolve()),
            }
        ],
    )
    record = {
        "status": "ok",
        "session_id": session_id,
        "user_audio": str(Path(args.user_audio).resolve()),
        "assistant_text": result["text"],
        "latency_ms": result["latency_ms"],
        "metadata": result["metadata"],
    }
    output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

