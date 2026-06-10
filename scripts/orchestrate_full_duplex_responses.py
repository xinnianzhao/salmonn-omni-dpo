#!/usr/bin/env python3
"""Generate raw multi-turn full-duplex responses for later DPO judging."""

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
from orchestrator.schemas import AudioRef, SessionRecord, TextResponse, TurnRecord
from orchestrator.topic_reader import iter_topics
from qwen_user_sim.client import QwenUserSimulator
from tts_interface.client import TTSClient


DEFAULT_TOPIC_FILE = "/lustre1/scratch/362/vsc36212/dataset/dpo/duplex_data/conv_natural_trivia_220k.json"
DEFAULT_PROMPT_AUDIO = "/scratch/leuven/362/vsc36212/projects/salmonn-omni/playground/data/cosy_prompt.wav"
DEFAULT_USER_PROMPT_AUDIO = PROJECT_ROOT / "asset/user_prompt.wav"
DEFAULT_ASSISTANT_PROMPT_AUDIO = PROJECT_ROOT / "asset/assis_prompt.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-file", default=DEFAULT_TOPIC_FILE)
    parser.add_argument("--topic", default=None, help="Use one literal topic instead of reading --topic-file.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=2, help="Number of independent episode rollouts per topic.")
    parser.add_argument("--output-jsonl", default="artifacts/dpo_raw/full_duplex_raw.jsonl")
    parser.add_argument("--artifact-dir", default="artifacts/dpo_raw")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--qwen-model", default="qwen3-32b-user-sim")
    parser.add_argument("--tts-base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--tts-mode", default="cross_lingual", choices=["zero_shot", "cross_lingual", "instruct2"])
    parser.add_argument("--tts-prompt-audio", default=None, help="Legacy fallback prompt for both user and assistant voices.")
    parser.add_argument("--tts-user-prompt-audio", default=str(DEFAULT_USER_PROMPT_AUDIO))
    parser.add_argument("--tts-assistant-prompt-audio", default=str(DEFAULT_ASSISTANT_PROMPT_AUDIO))
    parser.add_argument("--tts-prompt-text", default=None)
    parser.add_argument("--tts-language-tag", default="<|en|>")
    parser.add_argument("--full-duplex-repo", default="../salmonn-omni-dev")
    parser.add_argument("--full-duplex-python", default="python")
    parser.add_argument(
        "--full-duplex-extra-pythonpath",
        action="append",
        default=[],
        help="Additional import root for evaluate_hf.py. May be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--full-duplex-model-args-json", default=None)
    parser.add_argument("--full-duplex-data-args-json", default=None)
    parser.add_argument("--full-duplex-eval-args-json", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build records up to TTS, but skip full-duplex inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_prompt_args(args)
    artifact_dir = Path(args.artifact_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    qwen = QwenUserSimulator(base_url=args.qwen_base_url, model=args.qwen_model)
    tts = TTSClient(base_url=args.tts_base_url)
    full_duplex = FullDuplexEvalClient(
        repo_dir=Path(args.full_duplex_repo),
        python_bin=args.full_duplex_python,
        checkpoint_dir=Path(args.checkpoint_dir),
        work_dir=artifact_dir / "full_duplex_eval",
        device_id=args.device_id,
        model_args=load_json_arg(args.full_duplex_model_args_json),
        data_args=load_json_arg(args.full_duplex_data_args_json),
        eval_args=load_json_arg(args.full_duplex_eval_args_json),
        extra_pythonpaths=[Path(p) for p in args.full_duplex_extra_pythonpath] or None,
    )

    topic_iter = (
        [{"topic": args.topic, "source_id": "literal"}]
        if args.topic
        else iter_topics(args.topic_file, limit=args.limit, offset=args.offset)
    )
    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for topic_item in topic_iter:
            record = run_episode_pair(args, artifact_dir, qwen, tts, full_duplex, topic_item)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"Wrote episode pair {record['session_id']} status={record['status']}", flush=True)


def normalize_prompt_args(args: argparse.Namespace) -> None:
    fallback = args.tts_prompt_audio or DEFAULT_PROMPT_AUDIO
    if not args.tts_user_prompt_audio:
        args.tts_user_prompt_audio = fallback
    if not args.tts_assistant_prompt_audio:
        args.tts_assistant_prompt_audio = fallback


def run_episode_pair(
    args: argparse.Namespace,
    artifact_dir: Path,
    qwen: QwenUserSimulator,
    tts: TTSClient,
    full_duplex: FullDuplexEvalClient,
    topic_item: dict[str, str],
) -> dict:
    topic = topic_item["topic"]
    pair_session_id = f"{uuid.uuid4().hex[:12]}"
    rollouts = []
    errors = []

    for rollout_idx in range(args.rollouts):
        branch_id = branch_label(rollout_idx)
        rollout = run_rollout(
            args=args,
            artifact_dir=artifact_dir,
            qwen=qwen,
            tts=tts,
            full_duplex=full_duplex,
            topic_item=topic_item,
            pair_session_id=pair_session_id,
            branch_id=branch_id,
        )
        rollout_dict = rollout.to_dict()
        rollout_dict["branch_id"] = branch_id
        rollouts.append(rollout_dict)
        if rollout.status != "ok":
            errors.append({"branch_id": branch_id, "error": rollout.error})

    return {
        "schema_version": "full_duplex_dpo_episode_pair_v1",
        "session_id": pair_session_id,
        "topic": topic,
        "source": {"topic_file": args.topic_file, "source_id": topic_item.get("source_id")},
        "generation": {
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


def run_rollout(
    args: argparse.Namespace,
    artifact_dir: Path,
    qwen: QwenUserSimulator,
    tts: TTSClient,
    full_duplex: FullDuplexEvalClient,
    topic_item: dict[str, str],
    pair_session_id: str,
    branch_id: str,
) -> SessionRecord:
    topic = topic_item["topic"]
    session_id = f"{pair_session_id}_{branch_id}"
    session_dir = artifact_dir / session_id
    user_audio_dir = session_dir / "user_audio"
    assistant_audio_dir = session_dir / "assistant_audio"
    user_audio_dir.mkdir(parents=True, exist_ok=True)
    assistant_audio_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, str]] = []
    conv_v2_history: list[dict] = []
    turns: list[TurnRecord] = []
    try:
        for turn_id in range(1, args.turns + 1):
            user_result = qwen.generate_user_turn(topic=topic, history=history, turn_id=turn_id)
            user_text = user_result["utterance"]
            user_audio = synthesize_user_audio(args, tts, user_text, user_audio_dir, session_id, turn_id)

            conv_v2_history.append({"user": user_text, "user_path": user_audio.audio_path})
            if args.dry_run:
                assistant = {"text": "", "latency_ms": None, "metadata": {"dry_run": True}}
            else:
                assistant = full_duplex.generate_turn(
                    session_id=session_id,
                    turn_id=turn_id,
                    conv_turns=conv_v2_history,
                )

            assistant_text = assistant["text"]
            assistant_audio = None
            if assistant_text:
                assistant_audio = synthesize_assistant_history_audio(
                    args, tts, assistant_text, assistant_audio_dir, session_id, turn_id
                )
                conv_v2_history[-1]["assistant"] = assistant_text
                conv_v2_history[-1]["assistant_path"] = assistant_audio.audio_path

            turns.append(
                TurnRecord(
                    turn_id=turn_id,
                    user=TextResponse(
                        text=user_text,
                        audio=AudioRef(
                            path=user_audio.audio_path,
                            sample_rate=user_audio.sample_rate,
                            duration_sec=user_audio.duration_sec,
                            metadata=dict(user_result),
                        ),
                    ),
                    assistant_direct=TextResponse(
                        text=assistant_text,
                        audio=(
                            AudioRef(
                                path=assistant_audio.audio_path,
                                sample_rate=assistant_audio.sample_rate,
                                duration_sec=assistant_audio.duration_sec,
                                metadata={},
                            )
                            if assistant_audio
                            else None
                        ),
                        latency_ms=assistant.get("latency_ms"),
                        metadata=assistant.get("metadata", {}),
                    ),
                    assistant_candidates=[],
                )
            )
            history.extend([
                {"role": "user", "text": user_text},
                {"role": "assistant", "text": assistant_text},
            ])
        status = "ok"
        error = None
    except Exception as exc:
        status = "error"
        error = repr(exc)

    return SessionRecord(
        schema_version="full_duplex_dpo_raw_v1",
        session_id=session_id,
        topic=topic,
        source={"topic_file": args.topic_file, "source_id": topic_item.get("source_id")},
        generation={
            "turns_requested": args.turns,
            "pair_session_id": pair_session_id,
            "branch_id": branch_id,
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
            "note": "assistant_candidates is empty until evaluate_hf.py exposes non-greedy decoding.",
        },
        turns=turns,
        status=status,
        error=error,
    )


def synthesize_user_audio(args: argparse.Namespace, tts: TTSClient, text: str, output_dir: Path, session_id: str, turn_id: int):
    speak_text = f"{args.tts_language_tag}{text}" if args.tts_mode == "cross_lingual" else text
    return tts.synthesize(
        text=speak_text,
        output_dir=str(output_dir),
        mode=args.tts_mode,
        request_id=f"{session_id}_user_{turn_id:02d}",
        prompt_text=args.tts_prompt_text,
        prompt_audio_path=args.tts_user_prompt_audio,
    )


def synthesize_assistant_history_audio(
    args: argparse.Namespace,
    tts: TTSClient,
    text: str,
    output_dir: Path,
    session_id: str,
    turn_id: int,
):
    speak_text = f"{args.tts_language_tag}{text}" if args.tts_mode == "cross_lingual" else text
    return tts.synthesize(
        text=speak_text,
        output_dir=str(output_dir),
        mode=args.tts_mode,
        request_id=f"{session_id}_assistant_{turn_id:02d}",
        prompt_text=args.tts_prompt_text,
        prompt_audio_path=args.tts_assistant_prompt_audio,
    )


def branch_label(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return f"R{index + 1}"


def load_json_arg(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


if __name__ == "__main__":
    main()
