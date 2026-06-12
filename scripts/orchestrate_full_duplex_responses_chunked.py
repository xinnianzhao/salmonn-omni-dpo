#!/usr/bin/env python3
"""Generate episode pairs with chunked full-duplex subprocess batching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.full_duplex_client import FullDuplexEvalClient
from orchestrator.topic_reader import iter_topics
from full_duplex_service.client import FullDuplexServiceClient
from qwen_user_sim.client import QwenUserSimulator
from tts_interface.client import TTSClient


DEFAULT_TOPIC_FILE = "/lustre1/scratch/362/vsc36212/dataset/dpo/duplex_data/conv_natural_trivia_220k.json"
DEFAULT_USER_PROMPT_AUDIO = PROJECT_ROOT / "asset/user_prompt.wav"
DEFAULT_ASSISTANT_PROMPT_AUDIO = PROJECT_ROOT / "asset/assis_prompt.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-file", default=DEFAULT_TOPIC_FILE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4, help="Episode pairs per chunk.")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--output-jsonl", default="artifacts/dpo_chunked/episode_pairs.jsonl")
    parser.add_argument("--artifact-dir", default="artifacts/dpo_chunked")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--qwen-model", default="qwen3-32b-user-sim")
    parser.add_argument("--tts-base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--tts-mode", default="cross_lingual", choices=["zero_shot", "cross_lingual", "instruct2"])
    parser.add_argument("--tts-user-prompt-audio", default=str(DEFAULT_USER_PROMPT_AUDIO))
    parser.add_argument("--tts-assistant-prompt-audio", default=str(DEFAULT_ASSISTANT_PROMPT_AUDIO))
    parser.add_argument("--tts-prompt-text", default=None)
    parser.add_argument("--tts-language-tag", default="<|en|>")
    parser.add_argument("--full-duplex-repo", default="../salmonn-omni-dev")
    parser.add_argument("--full-duplex-python", default="python")
    parser.add_argument(
        "--full-duplex-base-url",
        default=None,
        help="Use a running full-duplex HTTP service instead of local subprocess calls.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--full-duplex-model-args-json", default=None)
    parser.add_argument("--full-duplex-data-args-json", default=None)
    parser.add_argument("--full-duplex-eval-args-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    artifact_dir = Path(args.artifact_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    qwen = QwenUserSimulator(base_url=args.qwen_base_url, model=args.qwen_model)
    tts = TTSClient(base_url=args.tts_base_url)
    if args.full_duplex_base_url:
        full_duplex = FullDuplexServiceAdapter(args.full_duplex_base_url)
    else:
        full_duplex = FullDuplexEvalClient(
            repo_dir=Path(args.full_duplex_repo),
            python_bin=args.full_duplex_python,
            checkpoint_dir=Path(args.checkpoint_dir),
            work_dir=artifact_dir / "full_duplex_eval",
            device_id=args.device_id,
            model_args=load_json_arg(args.full_duplex_model_args_json),
            data_args=load_json_arg(args.full_duplex_data_args_json),
            eval_args=load_json_arg(args.full_duplex_eval_args_json),
        )

    topics = list(iter_topics(args.topic_file, limit=args.limit, offset=args.offset))
    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for chunk_idx, start in enumerate(range(0, len(topics), args.chunk_size)):
            chunk_topics = topics[start : start + args.chunk_size]
            records = run_chunk(args, artifact_dir, qwen, tts, full_duplex, chunk_topics, args.offset + start, chunk_idx)
            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"wrote episode pair {record['session_id']} status={record['status']}", flush=True)
            out_f.flush()


def run_chunk(
    args: argparse.Namespace,
    artifact_dir: Path,
    qwen: QwenUserSimulator,
    tts: TTSClient,
    full_duplex: FullDuplexEvalClient,
    topic_items: list[dict[str, str]],
    episode_start_idx: int,
    chunk_idx: int,
) -> list[dict[str, Any]]:
    states = []
    for local_idx, topic_item in enumerate(topic_items):
        pair_session_id = f"ep{episode_start_idx + local_idx}"
        for rollout_idx in range(args.rollouts):
            branch_id = branch_label(rollout_idx)
            session_id = f"{pair_session_id}_{branch_id}"
            session_dir = artifact_dir / session_id
            states.append(
                {
                    "pair_session_id": pair_session_id,
                    "session_id": session_id,
                    "branch_id": branch_id,
                    "topic_item": topic_item,
                    "history": [],
                    "conv_v2_history": [],
                    "turns": [],
                    "status": "ok",
                    "error": None,
                    "user_audio_dir": session_dir / "user_audio",
                    "assistant_audio_dir": session_dir / "assistant_audio",
                }
            )
            states[-1]["user_audio_dir"].mkdir(parents=True, exist_ok=True)
            states[-1]["assistant_audio_dir"].mkdir(parents=True, exist_ok=True)

    for turn_id in range(1, args.turns + 1):
        batch_requests = []
        request_to_state = {}
        for state in states:
            if state["status"] != "ok":
                continue
            try:
                topic = state["topic_item"]["topic"]
                user_result = qwen.generate_user_turn(topic=topic, history=state["history"], turn_id=turn_id)
                user_text = user_result["utterance"]
                user_audio = synthesize_audio(
                    args=args,
                    tts=tts,
                    text=user_text,
                    output_dir=state["user_audio_dir"],
                    session_id=state["session_id"],
                    turn_id=turn_id,
                    speaker="user",
                )
                state["pending_user"] = {"text": user_text, "audio": audio_dict(user_audio), "metadata": user_result}
                state["conv_v2_history"].append({"user": user_text, "user_path": user_audio.audio_path})
                run_id = f"{state['session_id']}_turn{turn_id:02d}"
                batch_requests.append(
                    {
                        "session_id": state["session_id"],
                        "turn_id": turn_id,
                        "conv_turns": json.loads(json.dumps(state["conv_v2_history"], ensure_ascii=False)),
                        "run_id": run_id,
                    }
                )
                request_to_state[run_id] = state
            except Exception as exc:
                state["status"] = "error"
                state["error"] = repr(exc)

        if not batch_requests:
            continue

        batch_id = f"chunk{chunk_idx:04d}_turn{turn_id:02d}"
        try:
            assistant_results = full_duplex.generate_batch_turns(requests=batch_requests, batch_id=batch_id)
        except Exception as exc:
            for state in request_to_state.values():
                state["status"] = "error"
                state["error"] = repr(exc)
            continue

        for run_id, assistant in assistant_results.items():
            state = request_to_state[run_id]
            try:
                assistant_text = assistant["text"]
                assistant_audio = None
                if assistant_text:
                    assistant_audio = synthesize_audio(
                        args=args,
                        tts=tts,
                        text=assistant_text,
                        output_dir=state["assistant_audio_dir"],
                        session_id=state["session_id"],
                        turn_id=turn_id,
                        speaker="assistant",
                    )
                    state["conv_v2_history"][-1]["assistant"] = assistant_text
                    state["conv_v2_history"][-1]["assistant_path"] = assistant_audio.audio_path

                state["turns"].append(
                    {
                        "turn_id": turn_id,
                        "user": {
                            "text": state["pending_user"]["text"],
                            "audio": state["pending_user"]["audio"],
                            "latency_ms": None,
                            "metadata": state["pending_user"]["metadata"],
                        },
                        "assistant_direct": {
                            "text": assistant_text,
                            "audio": audio_dict(assistant_audio) if assistant_audio else None,
                            "latency_ms": assistant.get("latency_ms"),
                            "metadata": assistant.get("metadata", {}),
                        },
                        "assistant_candidates": [],
                    }
                )
                state["history"].extend(
                    [
                        {"role": "user", "text": state["pending_user"]["text"]},
                        {"role": "assistant", "text": assistant_text},
                    ]
                )
                state.pop("pending_user", None)
            except Exception as exc:
                state["status"] = "error"
                state["error"] = repr(exc)

    return build_pair_records(args, full_duplex, states)


def build_pair_records(args: argparse.Namespace, full_duplex: FullDuplexEvalClient, states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        by_pair.setdefault(state["pair_session_id"], []).append(state)

    records = []
    for pair_session_id, pair_states in by_pair.items():
        topic_item = pair_states[0]["topic_item"]
        rollouts = [session_record(args, full_duplex, state) for state in pair_states]
        errors = [
            {"branch_id": state["branch_id"], "error": state["error"]}
            for state in pair_states
            if state["status"] != "ok"
        ]
        records.append(
            {
                "schema_version": "full_duplex_dpo_episode_pair_v1",
                "session_id": pair_session_id,
                "topic": topic_item["topic"],
                "source": {"topic_file": args.topic_file, "source_id": topic_item.get("source_id")},
                "generation": {
                    "mode": getattr(full_duplex, "mode", "chunked_full_duplex_subprocess_batch"),
                    "chunk_size": args.chunk_size,
                    "turns_requested": args.turns,
                    "rollouts_requested": args.rollouts,
                    "qwen_base_url": args.qwen_base_url,
                    "qwen_model": args.qwen_model,
                    "tts_base_url": args.tts_base_url,
                    "tts_mode": args.tts_mode,
                    "tts_user_prompt_audio": args.tts_user_prompt_audio,
                    "tts_assistant_prompt_audio": args.tts_assistant_prompt_audio,
                    "full_duplex_repo": args.full_duplex_repo,
                    "checkpoint_dir": args.checkpoint_dir,
                    "full_duplex_task": full_duplex.task,
                    "full_duplex_template": full_duplex.template,
                    "pairing": "independent_rollouts_no_per_turn_branching",
                },
                "rollouts": rollouts,
                "judge": None,
                "status": "ok" if not errors else "error",
                "errors": errors,
            }
        )
    return records


def session_record(args: argparse.Namespace, full_duplex: FullDuplexEvalClient, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "full_duplex_dpo_raw_v1",
        "session_id": state["session_id"],
        "topic": state["topic_item"]["topic"],
        "source": {"topic_file": args.topic_file, "source_id": state["topic_item"].get("source_id")},
        "generation": {
            "turns_requested": args.turns,
            "pair_session_id": state["pair_session_id"],
            "branch_id": state["branch_id"],
            "qwen_base_url": args.qwen_base_url,
            "qwen_model": args.qwen_model,
            "tts_base_url": args.tts_base_url,
            "tts_mode": args.tts_mode,
            "tts_user_prompt_audio": args.tts_user_prompt_audio,
            "tts_assistant_prompt_audio": args.tts_assistant_prompt_audio,
            "full_duplex_repo": args.full_duplex_repo,
            "checkpoint_dir": args.checkpoint_dir,
            "full_duplex_task": full_duplex.task,
            "full_duplex_template": full_duplex.template,
            "note": "Full-duplex calls are batched per chunk and turn.",
        },
        "turns": state["turns"],
        "status": state["status"],
        "error": state["error"],
        "branch_id": state["branch_id"],
    }


def synthesize_audio(
    *,
    args: argparse.Namespace,
    tts: TTSClient,
    text: str,
    output_dir: Path,
    session_id: str,
    turn_id: int,
    speaker: str,
):
    prompt_audio = args.tts_user_prompt_audio if speaker == "user" else args.tts_assistant_prompt_audio
    speak_text = f"{args.tts_language_tag}{text}" if args.tts_mode == "cross_lingual" else text
    return tts.synthesize(
        text=speak_text,
        output_dir=str(output_dir),
        mode=args.tts_mode,
        request_id=f"{session_id}_{speaker}_{turn_id:02d}",
        prompt_text=args.tts_prompt_text,
        prompt_audio_path=prompt_audio,
    )


def audio_dict(response) -> dict[str, Any]:
    return {
        "path": response.audio_path,
        "sample_rate": response.sample_rate,
        "duration_sec": response.duration_sec,
        "metadata": dict(response.metadata or {}),
    }


def branch_label(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return f"R{index + 1}"


def load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


class FullDuplexServiceAdapter:
    task = "conv_v2"
    template = "default"
    mode = "chunked_full_duplex_api"

    def __init__(self, base_url: str):
        self.client = FullDuplexServiceClient(base_url=base_url)

    def generate_batch_turns(self, *, requests: list[dict[str, Any]], batch_id: str) -> dict[str, dict[str, Any]]:
        response = self.client.generate_batch(batch_id=batch_id, requests_=requests)
        normalized = {}
        for key, value in response.results.items():
            if hasattr(value, "model_dump"):
                normalized[key] = value.model_dump()
            else:
                normalized[key] = value.dict()
        return normalized


if __name__ == "__main__":
    main()
