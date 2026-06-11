#!/usr/bin/env python3
"""Extract compact judge inputs from raw episode-pair JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Raw episode_pairs.jsonl from orchestration.")
    parser.add_argument("--episode-output", default=None, help="Compact episode-pair judge file.")
    parser.add_argument("--last-turn-output", default=None, help="Compact last-turn judge file.")
    parser.add_argument("--last-turn", type=int, default=5, help="Target turn for last-turn judging.")
    parser.add_argument("--include-failed", action="store_true", help="Keep records whose raw status is not ok.")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rollout_by_branch(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for rollout in raw.get("rollouts", []):
        branch = rollout.get("branch_id") or rollout.get("session_id", "")[-1:]
        if branch:
            result[str(branch)] = rollout
    return result


def messages_from_turns(turns: list[dict[str, Any]], upto_turn: int | None = None) -> list[dict[str, str]]:
    messages = []
    for turn in turns:
        turn_id = int(turn.get("turn_id", len(messages) // 2 + 1))
        if upto_turn is not None and turn_id > upto_turn:
            break
        user_text = ((turn.get("user") or {}).get("text") or "").strip()
        assistant_text = ((turn.get("assistant_direct") or {}).get("text") or "").strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


def context_and_candidate(turns: list[dict[str, Any]], target_turn: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    context: list[dict[str, str]] = []
    candidate = {"role": "assistant", "content": ""}
    for turn in turns:
        turn_id = int(turn.get("turn_id", 0))
        user_text = ((turn.get("user") or {}).get("text") or "").strip()
        assistant_text = ((turn.get("assistant_direct") or {}).get("text") or "").strip()
        if turn_id < target_turn:
            if user_text:
                context.append({"role": "user", "content": user_text})
            if assistant_text:
                context.append({"role": "assistant", "content": assistant_text})
        elif turn_id == target_turn:
            if user_text:
                context.append({"role": "user", "content": user_text})
            candidate = {"role": "assistant", "content": assistant_text}
            break
    return context, candidate


def build_episode_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    rollouts = rollout_by_branch(raw)
    if "A" not in rollouts or "B" not in rollouts:
        return None
    return {
        "id": raw["session_id"],
        "pair_type": "episode",
        "topic": raw.get("topic", ""),
        "source_id": raw.get("source", "").get("source_id", ""),
        "branches": {
            branch: messages_from_turns(rollouts[branch].get("turns", []))
            for branch in ("A", "B")
        },
        "judge_instruction": (
            "Choose the better full conversation for a stable full-duplex assistant. "
            "Prefer factuality, relevance, multi-turn coherence, naturalness, and low hallucination. "
            "Return winner as A, B, or tie."
        ),
    }


def build_last_turn_record(raw: dict[str, Any], target_turn: int) -> dict[str, Any] | None:
    rollouts = rollout_by_branch(raw)
    if "A" not in rollouts or "B" not in rollouts:
        return None

    contexts = {}
    candidates = {}
    for branch in ("A", "B"):
        context, candidate = context_and_candidate(rollouts[branch].get("turns", []), target_turn)
        if not candidate["content"]:
            return None
        contexts[branch] = context
        candidates[branch] = candidate

    same_context = contexts["A"] == contexts["B"]
    return {
        "id": f"{raw['session_id']}_turn{target_turn}",
        "raw_session_id": raw["session_id"],
        "pair_type": "last_turn",
        "target_turn": target_turn,
        "topic": raw.get("topic", ""),
        "source": raw.get("source", {}),
        "same_context": same_context,
        "contexts": contexts,
        "candidates": candidates,
        "judge_instruction": (
            "Choose the better assistant response at the target turn. "
            "Use the provided context, and prefer factuality, relevance, coherence, and low hallucination. "
            "Return winner as A, B, or tie."
        ),
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    usable = [r for r in rows if args.include_failed or r.get("status") == "ok"]

    if args.episode_output:
        episode_records = [r for r in (build_episode_record(raw) for raw in usable) if r is not None]
        write_records(args.episode_output, episode_records)
        print(f"wrote {len(episode_records)} episode judge records to {args.episode_output}")

    if args.last_turn_output:
        last_turn_records = [
            r for r in (build_last_turn_record(raw, args.last_turn) for raw in usable) if r is not None
        ]
        write_records(args.last_turn_output, last_turn_records)
        print(f"wrote {len(last_turn_records)} last-turn judge records to {args.last_turn_output}")


if __name__ == "__main__":
    main()

