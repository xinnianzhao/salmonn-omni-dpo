#!/usr/bin/env python
from __future__ import annotations

import argparse

from tts_interface import TTSClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--text", default="I want to plan a weekend trip to Kyoto.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["zero_shot", "cross_lingual", "instruct2"], default="zero_shot")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-audio-path", required=True)
    parser.add_argument("--instruct-text")
    parser.add_argument("--request-id")
    args = parser.parse_args()

    client = TTSClient(base_url=args.base_url)
    response = client.synthesize(
        text=args.text,
        output_dir=args.output_dir,
        mode=args.mode,
        request_id=args.request_id,
        prompt_text=args.prompt_text,
        prompt_audio_path=args.prompt_audio_path,
        instruct_text=args.instruct_text,
    )
    if hasattr(response, "model_dump_json"):
        print(response.model_dump_json(indent=2))
    else:
        print(response.json(indent=2))


if __name__ == "__main__":
    main()
