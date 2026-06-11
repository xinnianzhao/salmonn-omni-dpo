#!/usr/bin/env python3
"""Merge judge decisions with raw episode pairs into SALMONN DPO pair files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl", required=True, help="Raw episode_pairs.jsonl from orchestration.")
    parser.add_argument("--judge-json", required=True, help="Claude judge output JSON or JSONL.")
    parser.add_argument("--episode-output", default=None, help="Output episode DPO pairs.")
    parser.add_argument("--last-turn-output", default=None, help="Output last-turn DPO pairs.")
    parser.add_argument("--skip-ties", action="store_true", default=True)
    return parser.parse_args()


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("data", "examples", "judgements", "judgments", "results"):
        if isinstance(data.get(key), list):
            return data[key]
    return [data]


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_winner(row: dict[str, Any]) -> str | None:
    value = (
        row.get("winner")
        or row.get("choice")
        or row.get("chosen_branch")
        or row.get("preferred")
        or row.get("label")
    )
    if isinstance(value, dict):
        value = value.get("branch") or value.get("winner")
    if value is None:
        return None
    value = str(value).strip().upper()
    if value in {"A", "B"}:
        return value
    if value in {"TIE", "EQUAL", "DRAW", "NONE"}:
        return "tie"
    return None


def judge_id(row: dict[str, Any]) -> str | None:
    return row.get("id") or row.get("pair_id") or row.get("session_id")


def rollout_by_branch(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for rollout in raw.get("rollouts", []):
        branch = rollout.get("branch_id") or rollout.get("session_id", "")[-1:]
        if branch:
            result[str(branch)] = rollout
    return result


def conv_v2_item_from_rollout(rollout: dict[str, Any], upto_turn: int | None = None) -> dict[str, Any]:
    conv = []
    for turn in rollout.get("turns", []):
        turn_id = int(turn.get("turn_id", 0))
        if upto_turn is not None and turn_id > upto_turn:
            break
        user = turn.get("user") or {}
        assistant = turn.get("assistant_direct") or {}
        user_audio = user.get("audio") or {}
        assistant_audio = assistant.get("audio") or {}
        if not user.get("text") or not user_audio.get("path"):
            continue
        item = {
            "user": user["text"],
            "user_path": user_audio["path"],
        }
        if assistant.get("text"):
            item["assistant"] = assistant["text"]
        if assistant_audio.get("path"):
            item["assistant_path"] = assistant_audio["path"]
        conv.append(item)
    return {"task": "conv_v2", "text": conv}


def build_raw_index(raw_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["session_id"]: row for row in raw_rows if row.get("session_id")}


def build_episode_pair(raw: dict[str, Any], judge: dict[str, Any], winner: str) -> dict[str, Any] | None:
    rollouts = rollout_by_branch(raw)
    loser = "B" if winner == "A" else "A"
    if winner not in rollouts or loser not in rollouts:
        return None
    return {
        "id": raw["session_id"],
        "pair_type": "episode",
        "chosen": conv_v2_item_from_rollout(rollouts[winner]),
        "rejected": conv_v2_item_from_rollout(rollouts[loser]),
        "judge": judge,
        "topic": raw.get("topic", ""),
        "source": raw.get("source", {}),
    }


def build_last_turn_pair(
    raw: dict[str, Any],
    judge: dict[str, Any],
    winner: str,
    target_turn: int,
) -> dict[str, Any] | None:
    rollouts = rollout_by_branch(raw)
    loser = "B" if winner == "A" else "A"
    if winner not in rollouts or loser not in rollouts:
        return None
    return {
        "id": f"{raw['session_id']}_turn{target_turn}",
        "pair_type": "last_turn",
        "chosen": conv_v2_item_from_rollout(rollouts[winner], upto_turn=target_turn),
        "rejected": conv_v2_item_from_rollout(rollouts[loser], upto_turn=target_turn),
        "judge": judge,
        "topic": raw.get("topic", ""),
        "source": raw.get("source", {}),
    }


def main() -> None:
    args = parse_args()
    raw_rows = read_json_or_jsonl(args.raw_jsonl)
    raw_index = build_raw_index(raw_rows)
    judge_rows = read_json_or_jsonl(args.judge_json)

    episode_pairs = []
    last_turn_pairs = []
    skipped = 0
    for judge in judge_rows:
        row_id = judge_id(judge)
        winner = normalize_winner(judge)
        if not row_id or winner is None or winner == "tie":
            skipped += 1
            continue

        pair_type = judge.get("pair_type")
        if pair_type == "last_turn" or "_turn" in row_id:
            raw_session_id = judge.get("raw_session_id") or row_id.split("_turn", 1)[0]
            raw = raw_index.get(raw_session_id)
            if raw is None:
                skipped += 1
                continue
            target_turn = int(judge.get("target_turn") or row_id.rsplit("_turn", 1)[1])
            pair = build_last_turn_pair(raw, judge, winner, target_turn)
            if pair is None:
                skipped += 1
            else:
                last_turn_pairs.append(pair)
        else:
            raw = raw_index.get(row_id)
            if raw is None:
                skipped += 1
                continue
            pair = build_episode_pair(raw, judge, winner)
            if pair is None:
                skipped += 1
            else:
                episode_pairs.append(pair)

    if args.episode_output:
        write_records(args.episode_output, episode_pairs)
        print(f"wrote {len(episode_pairs)} episode DPO pairs to {args.episode_output}")
    if args.last_turn_output:
        write_records(args.last_turn_output, last_turn_pairs)
        print(f"wrote {len(last_turn_pairs)} last-turn DPO pairs to {args.last_turn_output}")
    print(f"skipped {skipped} judge rows")


if __name__ == "__main__":
    main()

