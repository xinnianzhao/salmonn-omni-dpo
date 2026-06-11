#!/usr/bin/env python3
"""Serial 2-GPU episode-pair smoke orchestrator.

This keeps only one large module resident at a time:

1. Start Qwen vLLM on GPUs 0,1, generate one user utterance, stop Qwen.
2. Start TTS on GPU 0, synthesize user audio, stop TTS.
3. Run SALMONN-Omni on GPU 0 for the assistant response.
4. Start TTS on GPU 0, synthesize assistant-history audio, stop TTS.

It writes the same episode-pair JSONL shape as
``orchestrate_full_duplex_responses.py``, but is intentionally slow.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.full_duplex_client import FullDuplexEvalClient
from orchestrator.schemas import AudioRef, SessionRecord, TextResponse, TurnRecord
from orchestrator.topic_reader import iter_topics
from qwen_user_sim.client import QwenUserSimulator
from tts_interface.client import TTSClient


DEFAULT_TOPIC_FILE = "/lustre1/scratch/362/vsc36212/dataset/dpo/duplex_data/conv_natural_trivia_220k.json"
DEFAULT_USER_PROMPT_AUDIO = PROJECT_ROOT / "asset/user_prompt.wav"
DEFAULT_ASSISTANT_PROMPT_AUDIO = PROJECT_ROOT / "asset/assis_prompt.wav"


class ManagedService:
    def __init__(
        self,
        *,
        name: str,
        cmd: list[str],
        env: dict[str, str],
        log_path: Path,
        health_url: str,
        timeout_sec: int,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.env = env
        self.log_path = log_path
        self.health_url = health_url
        self.timeout_sec = timeout_sec
        self.proc: subprocess.Popen | None = None
        self._log_f = None

    def __enter__(self) -> "ManagedService":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_f = self.log_path.open("a", encoding="utf-8")
        self._log_f.write(f"\n===== starting {self.name}: {' '.join(self.cmd)} =====\n")
        self._log_f.flush()
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(PROJECT_ROOT),
            env=self.env,
            stdout=self._log_f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._wait_ready()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        deadline = time.time() + self.timeout_sec
        last_error = None
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} exited early with code {self.proc.returncode}; see {self.log_path}"
                )
            try:
                response = requests.get(self.health_url, timeout=5)
                if response.status_code < 500:
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for {self.name} at {self.health_url}: {last_error}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=45)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None
        if self._log_f is not None:
            self._log_f.close()
            self._log_f = None
        clear_gpu_processes()


def clear_gpu_processes() -> None:
    subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-file", default=DEFAULT_TOPIC_FILE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--artifact-dir", default="artifacts/dpo_episode_pair_serial_smoke")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--qwen-port", type=int, default=8001)
    parser.add_argument("--tts-port", type=int, default=8021)
    parser.add_argument("--qwen-env", default=str(PROJECT_ROOT / "envs/qwen-vllm-py312"))
    parser.add_argument("--tts-python", default="/scratch/leuven/362/vsc36212/playground/tts_eval/envs/cosyvoice/bin/python")
    parser.add_argument("--full-duplex-python", required=True)
    parser.add_argument("--checkpoint-dir", default="/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k")
    parser.add_argument("--full-duplex-repo", default="/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dev")
    parser.add_argument("--tts-mode", default="cross_lingual", choices=["zero_shot", "cross_lingual", "instruct2"])
    parser.add_argument("--tts-language-tag", default="<|en|>")
    parser.add_argument("--tts-user-prompt-audio", default=str(DEFAULT_USER_PROMPT_AUDIO))
    parser.add_argument("--tts-assistant-prompt-audio", default=str(DEFAULT_ASSISTANT_PROMPT_AUDIO))
    parser.add_argument("--full-duplex-model-args-json", default=str(PROJECT_ROOT / "configs/full_duplex_default_model_args.json"))
    parser.add_argument("--full-duplex-data-args-json", default=str(PROJECT_ROOT / "configs/full_duplex_default_data_args.json"))
    parser.add_argument("--full-duplex-eval-args-json", default=str(PROJECT_ROOT / "configs/full_duplex_default_eval_args.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else artifact_dir / "episode_pairs.jsonl"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    full_duplex = FullDuplexEvalClient(
        repo_dir=Path(args.full_duplex_repo),
        python_bin=args.full_duplex_python,
        checkpoint_dir=Path(args.checkpoint_dir),
        work_dir=artifact_dir / "full_duplex_eval",
        device_id=0,
        model_args=load_json_arg(args.full_duplex_model_args_json),
        data_args=load_json_arg(args.full_duplex_data_args_json),
        eval_args=load_json_arg(args.full_duplex_eval_args_json),
    )

    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for episode_idx, topic_item in enumerate(iter_topics(args.topic_file, limit=args.limit, offset=args.offset)):
            record = run_episode_pair(args, artifact_dir, full_duplex, topic_item, episode_idx)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"wrote episode pair {record['session_id']} status={record['status']}", flush=True)


def run_episode_pair(
    args: argparse.Namespace,
    artifact_dir: Path,
    full_duplex: FullDuplexEvalClient,
    topic_item: dict[str, str],
    episode_idx: int,
) -> dict[str, Any]:
    pair_session_id = f"ep{episode_idx}"
    rollouts = []
    errors = []
    for rollout_idx in range(args.rollouts):
        branch_id = branch_label(rollout_idx)
        rollout = run_rollout(args, artifact_dir, full_duplex, topic_item, pair_session_id, branch_id)
        rollout_dict = rollout.to_dict()
        rollout_dict["branch_id"] = branch_id
        rollouts.append(rollout_dict)
        if rollout.status != "ok":
            errors.append({"branch_id": branch_id, "error": rollout.error})
    return {
        "schema_version": "full_duplex_dpo_episode_pair_v1",
        "session_id": pair_session_id,
        "topic": topic_item["topic"],
        "source": {"topic_file": args.topic_file, "source_id": topic_item.get("source_id")},
        "generation": {
            "mode": "serial_2gpu_episode_pair_smoke",
            "turns_requested": args.turns,
            "rollouts_requested": args.rollouts,
            "checkpoint_dir": args.checkpoint_dir,
            "tts_user_prompt_audio": args.tts_user_prompt_audio,
            "tts_assistant_prompt_audio": args.tts_assistant_prompt_audio,
            "pairing": "independent_rollouts_no_per_turn_branching",
            "note": "Serial mode starts/stops Qwen and TTS around each stage to avoid concurrent GPU residency.",
        },
        "rollouts": rollouts,
        "judge": None,
        "status": "ok" if not errors else "error",
        "errors": errors,
    }


def run_rollout(
    args: argparse.Namespace,
    artifact_dir: Path,
    full_duplex: FullDuplexEvalClient,
    topic_item: dict[str, str],
    pair_session_id: str,
    branch_id: str,
) -> SessionRecord:
    session_id = f"{pair_session_id}_{branch_id}"
    session_dir = artifact_dir / session_id
    user_audio_dir = session_dir / "user_audio"
    assistant_audio_dir = session_dir / "assistant_audio"
    history: list[dict[str, str]] = []
    conv_v2_history: list[dict[str, Any]] = []
    turns: list[TurnRecord] = []
    status = "ok"
    error = None

    try:
        for turn_id in range(1, args.turns + 1):
            with make_qwen_service(args, artifact_dir, session_id, turn_id):
                qwen_client = QwenUserSimulator(
                    base_url=f"http://127.0.0.1:{args.qwen_port}/v1",
                    model="qwen3-32b-user-sim",
                )
                user_result = qwen_client.generate_user_turn(
                    topic=topic_item["topic"],
                    history=history,
                    turn_id=turn_id,
                )
            user_text = user_result["utterance"]

            with make_tts_service(args, artifact_dir, session_id, turn_id, "user"):
                tts_client = TTSClient(base_url=f"http://127.0.0.1:{args.tts_port}")
                user_audio = tts_client.synthesize(
                    text=format_tts_text(args, user_text),
                    output_dir=str(user_audio_dir),
                    mode=args.tts_mode,
                    request_id=f"{session_id}_user_{turn_id:02d}",
                    prompt_audio_path=args.tts_user_prompt_audio,
                )

            conv_v2_history.append({"user": user_text, "user_path": user_audio.audio_path})
            assistant = full_duplex.generate_turn(
                session_id=session_id,
                turn_id=turn_id,
                conv_turns=conv_v2_history,
            )
            assistant_text = assistant["text"]
            if not assistant_text:
                raise RuntimeError(f"empty assistant response at turn {turn_id}")

            with make_tts_service(args, artifact_dir, session_id, turn_id, "assistant"):
                tts_client = TTSClient(base_url=f"http://127.0.0.1:{args.tts_port}")
                assistant_audio = tts_client.synthesize(
                    text=format_tts_text(args, assistant_text),
                    output_dir=str(assistant_audio_dir),
                    mode=args.tts_mode,
                    request_id=f"{session_id}_assistant_{turn_id:02d}",
                    prompt_audio_path=args.tts_assistant_prompt_audio,
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
                        audio=AudioRef(
                            path=assistant_audio.audio_path,
                            sample_rate=assistant_audio.sample_rate,
                            duration_sec=assistant_audio.duration_sec,
                        ),
                        latency_ms=assistant.get("latency_ms"),
                        metadata=assistant.get("metadata", {}),
                    ),
                )
            )
            history.extend([
                {"role": "user", "text": user_text},
                {"role": "assistant", "text": assistant_text},
            ])
    except Exception as exc:
        status = "error"
        error = repr(exc)

    return SessionRecord(
        schema_version="full_duplex_dpo_raw_v1",
        session_id=session_id,
        topic=topic_item["topic"],
        source={"topic_file": args.topic_file, "source_id": topic_item.get("source_id")},
        generation={
            "mode": "serial_2gpu_episode_pair_smoke",
            "pair_session_id": pair_session_id,
            "branch_id": branch_id,
            "turns_requested": args.turns,
            "checkpoint_dir": args.checkpoint_dir,
        },
        turns=turns,
        status=status,
        error=error,
    )


def make_qwen_service(args: argparse.Namespace, artifact_dir: Path, session_id: str, turn_id: int) -> ManagedService:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("QWEN_CUDA_VISIBLE_DEVICES", "0,1"),
            "PORT": str(args.qwen_port),
            "TENSOR_PARALLEL_SIZE": "2",
            "MAX_MODEL_LEN": os.environ.get("QWEN_MAX_MODEL_LEN", "8192"),
            "MAX_NUM_BATCHED_TOKENS": os.environ.get("QWEN_MAX_NUM_BATCHED_TOKENS", "8192"),
        }
    )
    return ManagedService(
        name="qwen",
        cmd=["bash", str(PROJECT_ROOT / "scripts/start_qwen3_32b_vllm.sh")],
        env=env,
        log_path=artifact_dir / "service_logs" / f"{session_id}_turn{turn_id:02d}_qwen.log",
        health_url=f"http://127.0.0.1:{args.qwen_port}/v1/models",
        timeout_sec=1800,
    )


def make_tts_service(
    args: argparse.Namespace,
    artifact_dir: Path,
    session_id: str,
    turn_id: int,
    stage: str,
) -> ManagedService:
    cache_dir = artifact_dir / ".cache" / f"{session_id}_turn{turn_id:02d}_{stage}"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("TTS_CUDA_VISIBLE_DEVICES", "0"),
            "PYTHONPATH": f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}",
            "COSYVOICE_REPO_DIR": os.environ.get(
                "COSYVOICE_REPO_DIR",
                "/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo",
            ),
            "COSYVOICE_MODEL_DIR": os.environ.get(
                "COSYVOICE_MODEL_DIR",
                "/lustre1/project/stg_00197/projects/salmonn-omni/playground/tts/CosyVoice2-0.5B",
            ),
            "XDG_CACHE_HOME": str(cache_dir),
            "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
            "NUMBA_CACHE_DIR": str(cache_dir / "numba"),
        }
    )
    return ManagedService(
        name="tts",
        cmd=[
            args.tts_python,
            "-m",
            "uvicorn",
            "tts_interface.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.tts_port),
        ],
        env=env,
        log_path=artifact_dir / "service_logs" / f"{session_id}_turn{turn_id:02d}_{stage}_tts.log",
        health_url=f"http://127.0.0.1:{args.tts_port}/health",
        timeout_sec=900,
    )


def format_tts_text(args: argparse.Namespace, text: str) -> str:
    return f"{args.tts_language_tag}{text}" if args.tts_mode == "cross_lingual" else text


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


if __name__ == "__main__":
    main()
