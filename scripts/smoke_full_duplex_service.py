#!/usr/bin/env python3
"""Smoke test the full-duplex HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from full_duplex_service.client import FullDuplexServiceClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8031")
    parser.add_argument("--user-audio", default=str(PROJECT_ROOT / "asset/user_prompt.wav"))
    parser.add_argument("--output-json", default="artifacts/full_duplex_service_smoke/result.json")
    parser.add_argument("--session-id", default="api_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = FullDuplexServiceClient(base_url=args.base_url, timeout_sec=3600)
    health = client.health()
    result = client.generate_turn(
        session_id=args.session_id,
        turn_id=1,
        run_id=f"{args.session_id}_turn01",
        conv_turns=[
            {
                "user": "Please briefly introduce the Outer Banks in North Carolina.",
                "user_path": str(Path(args.user_audio).resolve()),
            }
        ],
    )
    record = {
        "status": "ok",
        "health": health,
        "session_id": args.session_id,
        "assistant_text": result.text,
        "latency_ms": result.latency_ms,
        "metadata": result.metadata,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
