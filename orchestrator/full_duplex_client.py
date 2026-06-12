"""Subprocess client for SALMONN-Omni full-duplex evaluation.

The SALMONN-Omni repo currently exposes generation through ``evaluate_hf.py``.
This client keeps that codebase as an independent checkout and drives it by
writing a one-item conversation dataset, running the evaluator, then parsing the
result JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class FullDuplexEvalError(RuntimeError):
    """Raised when the full-duplex evaluator subprocess fails."""


@dataclass
class FullDuplexEvalClient:
    repo_dir: Path = Path("../salmonn-omni-dev")
    python_bin: str = "python"
    checkpoint_dir: Path = Path(
        "/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k"
    )
    work_dir: Path = Path("artifacts/full_duplex_eval")
    device_id: int = 0
    timeout_sec: int = 1800
    model_args: dict[str, Any] = field(default_factory=dict)
    data_args: dict[str, Any] = field(default_factory=dict)
    eval_args: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    task: str = "conv_v2"
    template: str = "default"
    extra_pythonpaths: list[Path] | None = field(
        default_factory=lambda: [
            Path("/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/models"),
            Path("/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/third_party/Matcha-TTS"),
        ]
    )

    def __post_init__(self) -> None:
        self.repo_dir = self.repo_dir.resolve()
        self.checkpoint_dir = self.checkpoint_dir.resolve()
        self.work_dir = self.work_dir.resolve()
        if self.extra_pythonpaths is None:
            self.extra_pythonpaths = [
                Path("/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/models"),
                Path("/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/third_party/Matcha-TTS"),
            ]
        self.extra_pythonpaths = [p.resolve() for p in self.extra_pythonpaths]

    def generate_turn(
        self,
        *,
        session_id: str,
        turn_id: int,
        conv_turns: list[dict[str, Any]],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate the assistant response for one target turn.

        For ``task="conv_v2", template="default"``, ``conv_turns`` is a list
        of paired turn records. Previous turns contain ``user_path``,
        ``assistant``, and ``assistant_path``. The target turn needs only
        ``user_path``.
        """
        run_id = run_id or f"{session_id}_turn{turn_id:02d}"
        results = self.generate_batch_turns(
            requests=[
                {
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "conv_turns": conv_turns,
                    "run_id": run_id,
                }
            ],
            batch_id=run_id,
        )
        return results[run_id]

    def generate_batch_turns(
        self,
        *,
        requests: list[dict[str, Any]],
        batch_id: str,
    ) -> dict[str, dict[str, Any]]:
        """Generate assistant responses for many same-turn sessions.

        ``evaluate_hf.py`` accepts a dataset containing multiple items, but
        only one global ``--num_turn``. Therefore every request in a batch must
        target the same turn id.
        """
        self._validate()
        if not requests:
            return {}

        turn_ids = {int(req["turn_id"]) for req in requests}
        if len(turn_ids) != 1:
            raise FullDuplexEvalError(f"Batch {batch_id} has mixed turn ids: {sorted(turn_ids)}")
        turn_id = turn_ids.pop()

        run_dir = self.work_dir / "batches" / batch_id
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = []
        item_ids = []
        for req in requests:
            item_id = req.get("run_id") or f"{req['session_id']}_turn{int(req['turn_id']):02d}"
            item_ids.append(item_id)
            dataset.append({"id": item_id, "task": self.task, "text": req["conv_turns"]})
        dataset_path = input_dir / f"{batch_id}.json"
        dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = self._build_command(dataset_path=dataset_path, output_dir=output_dir, num_turn=turn_id)
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_dir),
            env=self._build_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_sec,
            check=False,
        )
        latency_ms = (time.time() - started) * 1000.0

        (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        (run_dir / "command.json").write_text(
            json.dumps({"cmd": cmd, "cwd": str(self.repo_dir)}, indent=2),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise FullDuplexEvalError(
                f"evaluate_hf.py failed with code {proc.returncode}; see {run_dir}/stderr.txt"
            )

        result_path = output_dir / f"{dataset_path.stem}_0.json"
        if not result_path.exists():
            raise FullDuplexEvalError(f"Expected evaluator output not found: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        missing = [item_id for item_id in item_ids if item_id not in result]
        if missing:
            raise FullDuplexEvalError(f"Evaluator output missing item ids {missing}: {result_path}")

        per_item_latency_ms = latency_ms / max(len(item_ids), 1)
        return {
            item_id: {
                "text": str(result[item_id]).strip(),
                "latency_ms": per_item_latency_ms,
                "metadata": {
                    "run_dir": str(run_dir),
                    "dataset_path": str(dataset_path),
                    "result_path": str(result_path),
                    "command": cmd,
                    "batch_id": batch_id,
                    "batch_size": len(item_ids),
                    "batch_latency_ms": latency_ms,
                },
            }
            for item_id in item_ids
        }

    def _validate(self) -> None:
        if not (self.repo_dir / "evaluate_hf.py").exists():
            raise FullDuplexEvalError(f"evaluate_hf.py not found under {self.repo_dir}")
        if not self.checkpoint_dir.exists():
            raise FullDuplexEvalError(f"Checkpoint directory not found: {self.checkpoint_dir}")

    def _build_command(self, *, dataset_path: Path, output_dir: Path, num_turn: int) -> list[str]:
        model_args = {
            "model_name_or_path": "${HF_HOME}/hub/models--meta-llama--Meta-Llama-3-8B-Instruct",
            "ckpt_path": str(self.checkpoint_dir),
            "attn_implementation": "flash_attention_2",
            "backbone": "llama3",
            "speech_synthesizer_type": "none",
            "add_assist_stream": True,
            "simple_mode": True,
            "add_eor_token": False,
            "add_clause_sep_token": False,
            "add_end_of_speech": True,
            **self.model_args,
        }
        model_args["model_name_or_path"] = str(_resolve_hf_cache_model_path(model_args["model_name_or_path"]))
        data_args = {
            "dataset_dir": str(dataset_path),
            "template": self.template,
            "num_audio_embs": 1,
            "num_text_tokens": 1,
            "delta": 0.08,
            "max_audio_len": 125,
            "turn_wise_encode": False,
            "diff_shift": True,
            "flexible_conv": True,
            "same_prompt": True,
            "reason_before_speak": False,
            **self.data_args,
        }
        eval_args = {
            "num_turn": num_turn,
            "single_gpu": True,
            "device_id": self.device_id,
            "eval_output_dir": str(output_dir),
            "echo_ratio": 0.0,
            **self.eval_args,
        }
        cmd = [self.python_bin, str(self.repo_dir / "evaluate_hf.py")]
        for args in (model_args, data_args, eval_args):
            for key, value in args.items():
                cmd.extend(_hf_arg(key, value))
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env)
        project_root = Path(__file__).resolve().parents[1]
        pythonpath = [str(project_root / "compat"), str(self.repo_dir)]
        pythonpath.extend(str(path) for path in self.extra_pythonpaths if path.exists())
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        return env


def _resolve_hf_cache_model_path(path_or_model_id: Any) -> str | Path:
    """Resolve HF cache repo roots to a concrete snapshot directory.

    Transformers accepts either a Hub repo id or a local model directory. The
    cache root `.../hub/models--org--repo` is not directly loadable; the usable
    files live under `snapshots/<revision>`.
    """

    raw = os.path.expandvars(os.path.expanduser(str(path_or_model_id)))
    path = Path(raw)
    if not path.exists():
        return raw
    if (path / "config.json").exists() or (path / "tokenizer_config.json").exists():
        return path

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.is_dir():
        return path

    ref_path = path / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = snapshots_dir / revision
        if snapshot.exists():
            return snapshot

    snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if snapshots:
        return snapshots[0]
    return path


def _hf_arg(key: str, value: Any) -> list[str]:
    if isinstance(value, bool):
        return [f"--{key}", str(value).lower()]
    if isinstance(value, (list, tuple)):
        return [f"--{key}", *[str(v) for v in value]]
    if value is None:
        return []
    return [f"--{key}", str(value)]
