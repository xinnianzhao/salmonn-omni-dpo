#!/usr/bin/env python
from __future__ import annotations

import argparse

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3-32b-user-sim")
    parser.add_argument("--prompt", default="Start a conversation about planning a weekend trip to Kyoto.")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are simulating a spoken user. Produce only the next user utterance. "
                    "Do not answer as an assistant. Output plain text only, with no thinking text."
                ),
            },
            {"role": "user", "content": args.prompt},
        ],
        temperature=0.6,
        top_p=0.9,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(response.choices[0].message.content.strip())


if __name__ == "__main__":
    main()
