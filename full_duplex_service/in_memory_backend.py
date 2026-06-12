from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import torchaudio
from lhotse import Fbank, FbankConfig

from orchestrator.full_duplex_client import _resolve_hf_cache_model_path


class InMemoryFullDuplexError(RuntimeError):
    pass


def _repo_to_path(repo_dir: str | Path) -> Path:
    repo_path = Path(repo_dir).resolve()
    if not (repo_path / "utils.py").exists():
        raise FileNotFoundError(f"SALMONN repo not found or invalid: {repo_path}")
    extra_paths = [
        Path(__file__).resolve().parents[1] / "compat",
        repo_path,
        Path("/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/models"),
        Path("/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/third_party/Matcha-TTS"),
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_path


class FlexibleArgs:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _default_model_args() -> dict[str, Any]:
    return {
        "model_name_or_path": "${HF_HOME}/hub/models--meta-llama--Meta-Llama-3-8B-Instruct",
        "attn_implementation": "flash_attention_2",
        "autocast_dtype": "bfloat16",
        "backbone": "llama3",
        "backbone_setting": ["lora", "32", "32", "0.1"],
        "speech_encoder_type": "transformer_v2",
        "speech_encoder_trainable": False,
        "speech_enc_llm_proj_setting": ["4", "4096"],
        "speech_synthesizer_type": "none",
        "tts_loss_weight": 0.5,
        "add_assist_stream": True,
        "simple_mode": True,
        "lazy_encode": True,
        "enc_path": None,
        "ckpt_path": None,
        "add_eor_token": False,
        "add_clause_sep_token": False,
        "add_end_of_speech": True,
        "enable_latent_reasoning": False,
        "use_edl_loss": False,
        "latent_distribution": "",
        "add_duplex_head": False,
        "duplex_head_transformer": False,
        "duplex_detach_hidden": False,
        "add_reason_level_head": False,
        "latent_target_norm": False,
        "latent_target_fixed_std": False,
        "latent_target_interpolate": False,
        "latent_target_interpolate_actual_count": False,
        "latent_pad_with_ellipsis": False,
        "latent_reason_use_question": False,
        "latent_asr": False,
        "latent_head_independent_mlps": False,
        "latent_head_layer": -1,
        "latent_accumulate": False,
        "latent_accumulate_interleave": False,
        "use_init_latent_embed": False,
        "latent_freq": 1,
        "latent_split_n": 1,
        "latent_accumulate_add_to_last_input": False,
        "latent_accumulate_add_to_next_input": False,
        "latent_accumulate_lm_ce_loss": False,
        "decode_latent_token": False,
        "add_latent_token": False,
    }


def _default_data_args() -> dict[str, Any]:
    return {
        "dataset_dir": "",
        "context_independent_barge_in_data": None,
        "backchanneling_data": None,
        "max_seq_len": 4500,
        "max_audio_len": 125,
        "delta": 0.08,
        "template": "default",
        "num_audio_embs": 1,
        "num_text_tokens": 1,
        "think_loss_weight": 0.1,
        "shift_loss_weight": 1.0,
        "listen_loss_weight": 0.1,
        "speak_loss_weight": 0.1,
        "turn_wise_encode": False,
        "barge_in_ratio": 0.4,
        "diff_shift": True,
        "rev_noise_ratio": 0.0,
        "random_echo": False,
        "random_echo_v2": False,
        "simple_mode_echo_ratio": 0.0,
        "flexible_conv": True,
        "sim_topic_shift": 0.0,
        "use_multi_speaker_prompt": False,
        "num_reason_tokens": 10,
        "same_prompt": True,
        "reason_question_data": None,
        "reason_loss_weight": 0.0,
        "reason_level_loss_weight": 0.0,
        "latent_accumulate_lm_ce_loss_weight": 1.0,
        "reason_before_speak": False,
        "latent_explicit_reason": False,
        "gt_latent": False,
        "no_latent": False,
        "disable_clause_qa_reason": False,
        "use_text_question": False,
        "latent_pad_loss_weight": 1.0,
        "latent_dirichlet_topk": 64,
        "latent_dirichlet_alpha_eps": 1e-4,
        "latent_dirichlet_embed_loss_weight": 1.0,
        "latent_dirichlet_dm_loss_weight": 0.25,
        "latent_dirichlet_loss_variant": "legacy",
        "latent_dirichlet_cover_loss_weight": 0.25,
        "latent_dirichlet_scale_loss_weight": 0.01,
    }


def _default_eval_args() -> dict[str, Any]:
    return {
        "seed": 42,
        "oracle_start": False,
        "verbose": False,
        "conv_dynamic": "none",
        "echo_ratio": 0.0,
        "single_gpu": True,
        "device_id": 0,
    }


class InMemoryFullDuplexGenerator:
    """In-process conv_v2/default generator ported from evaluate_hf.py.

    This intentionally supports the current DPO generation path only:
    `task=conv_v2`, `template=default`, `speech_synthesizer_type=none`.
    """

    def __init__(
        self,
        *,
        repo_dir: str | Path,
        checkpoint_dir: str | Path,
        model_args: dict[str, Any],
        data_args: dict[str, Any],
        eval_args: dict[str, Any],
        device_id: int = 0,
    ) -> None:
        _repo_to_path(repo_dir)
        from utils import load_model, setup_logger

        self.model_args = FlexibleArgs(**_default_model_args())
        self.model_args.__dict__.update(model_args)
        self.data_args = FlexibleArgs(**_default_data_args())
        self.data_args.__dict__.update(data_args)
        self.eval_args = FlexibleArgs(**_default_eval_args())
        self.eval_args.__dict__.update(eval_args)
        self.model_args.model_name_or_path = str(_resolve_hf_cache_model_path(self.model_args.model_name_or_path))
        self.model_args.ckpt_path = str(Path(checkpoint_dir).resolve())
        self.model_args.speech_synthesizer_type = "none"
        self.data_args.template = "default"
        self.data_args.lhotse_fbank = self.model_args.speech_encoder_type in (
            "zipformer2",
            "transformer",
            "transformer_v2",
        )
        self.eval_args.conv_dynamic = "none"
        self.eval_args.single_gpu = True
        self.eval_args.device_id = device_id

        setup_logger()
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        self.tokenizer, self.model = load_model(self.model_args, self.data_args)
        self.model.to(self.device).to(torch.bfloat16).eval()
        self.num_audio_embs = (
            self.data_args.num_audio_embs * 2
            if (self.model_args.add_assist_stream and not self.model_args.get("simple_mode", False))
            else self.data_args.num_audio_embs
        )
        if self.data_args.lhotse_fbank:
            self.fbank = Fbank(FbankConfig(num_mel_bins=128))
        else:
            import kaldifeat

            opts = kaldifeat.FbankOptions()
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = 16000
            opts.mel_opts.num_bins = 80
            opts.mel_opts.high_freq = -400
            self.fbank = kaldifeat.Fbank(opts)
        self._lock = Lock()

    def generate_batch_turns(self, requests: list[dict[str, Any]], batch_id: str) -> dict[str, dict[str, Any]]:
        if not requests:
            return {}
        turn_ids = {int(req["turn_id"]) for req in requests}
        if len(turn_ids) != 1:
            raise InMemoryFullDuplexError(f"Batch {batch_id} has mixed turn ids: {sorted(turn_ids)}")

        results = {}
        with self._lock, torch.inference_mode():
            for req in requests:
                run_id = req.get("run_id") or f"{req['session_id']}_turn{int(req['turn_id']):02d}"
                start = time.time()
                text = self.generate_conv_v2_turn(req["conv_turns"], int(req["turn_id"]))
                results[run_id] = {
                    "text": text,
                    "latency_ms": (time.time() - start) * 1000.0,
                    "metadata": {
                        "backend": "in_memory_conv_v2",
                        "batch_id": batch_id,
                        "run_id": run_id,
                    },
                }
        return results

    def generate_conv_v2_turn(self, conv: list[dict[str, Any]], num_turn: int) -> str:
        if self.data_args.template != "default":
            raise InMemoryFullDuplexError("in-memory backend currently supports template=default only")
        if self.model_args.backbone != "llama3":
            raise InMemoryFullDuplexError("in-memory backend currently supports llama3 only")
        if self.data_args.turn_wise_encode:
            raise InMemoryFullDuplexError("turn_wise_encode is not supported for conv_v2 inference")

        target_turn_idx = num_turn - 1
        if target_turn_idx < 0 or target_turn_idx >= len(conv):
            raise InMemoryFullDuplexError("num_turn exceeds conversation length")

        env_stream: list[torch.Tensor] = []
        assist_stream: list[torch.Tensor] = []
        n_time_blocks: list[int] = []
        fs = 16000
        if self.model_args.lazy_encode:
            self.model.need_encode = True

        input_ids = self._tokenize(
            "<|begin_of_text|>"
            + self.tokenizer.audio_pad_token * self.num_audio_embs
            + "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
        )

        def add_user_audio(turn: dict[str, Any], add_input_tokens: bool = True):
            nonlocal input_ids, fs
            audio, fs = self._load_audio(turn["user_path"])
            n_time_block = (
                math.ceil((audio.shape[1] + sum(a.shape[1] for a in env_stream)) / fs / self.data_args.delta)
                - sum(n_time_blocks)
            )
            n_time_blocks.append(n_time_block)
            env_stream.append(audio)
            assist_stream.append(torch.zeros_like(audio))
            if add_input_tokens:
                for _ in range(n_time_block - 1):
                    input_ids = torch.cat(
                        (
                            input_ids,
                            self._tokenize(self.tokenizer.audio_pad_token * self.num_audio_embs + self.tokenizer.think_token),
                        )
                    )
            return n_time_block

        def add_assistant_turn(turn: dict[str, Any]):
            nonlocal input_ids, fs
            if not turn.get("assistant"):
                raise InMemoryFullDuplexError("assistant text is empty in history turn")
            audio, fs = self._load_audio(turn["assistant_path"])
            env_stream.append(audio * float(self.eval_args.echo_ratio))
            assist_stream.append(audio)

            answer_ids = self._tokenize(turn["assistant"] + "<|eot_id|>")
            n_time_block = math.floor(sum(a.shape[1] for a in env_stream) / fs / self.data_args.delta) - sum(n_time_blocks)
            n_time_blocks.append(n_time_block)

            text_idx = 0
            time_idx = 0
            while text_idx < len(answer_ids):
                cur_text = self.tokenizer.audio_pad_token * self.num_audio_embs
                if text_idx == 0:
                    cur_text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    time_idx -= 1
                cur_ids = self._tokenize(cur_text)
                input_ids = torch.cat(
                    (
                        input_ids,
                        cur_ids,
                        answer_ids[max(text_idx - 1, 0) : text_idx + self.data_args.num_text_tokens - 1],
                    )
                )
                text_idx += self.data_args.num_text_tokens
                time_idx += 1

            n_used_time_block = time_idx
            if n_used_time_block >= n_time_block:
                raise InMemoryFullDuplexError("too long answer")

            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if input_ids[-1] == eot_id:
                input_ids = input_ids[:-1]

            for time_idx in range(n_used_time_block, n_time_block):
                is_final_speak_block = time_idx == n_time_block - 1
                if self.model_args.get("add_end_of_speech", False) and is_final_speak_block:
                    cur_text = self.tokenizer.audio_pad_token * self.num_audio_embs + "<|end_of_speech|>" + self.tokenizer.think_token
                elif time_idx == n_used_time_block:
                    cur_text = self.tokenizer.audio_pad_token * self.num_audio_embs + "<|eot_id|>"
                else:
                    cur_text = self.tokenizer.audio_pad_token * self.num_audio_embs + self.tokenizer.think_token
                input_ids = torch.cat((input_ids, self._tokenize(cur_text)))

        def add_user_header():
            nonlocal input_ids
            input_ids = torch.cat(
                (
                    input_ids,
                    self._tokenize(
                        self.tokenizer.audio_pad_token * self.num_audio_embs
                        + "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
                    ),
                )
            )

        for turn_idx in range(target_turn_idx):
            add_user_audio(conv[turn_idx], add_input_tokens=True)
            add_assistant_turn(conv[turn_idx])
            add_user_header()

        pad_stream = torch.zeros_like(torch.cat(env_stream, dim=1)) if env_stream else None
        n_time_block = add_user_audio(conv[target_turn_idx], add_input_tokens=False)

        env_stream.append(torch.zeros(1, fs * 120))
        assist_stream.append(torch.zeros(1, fs * 120))
        env_tensor = torch.cat(env_stream, dim=1)
        assist_tensor = torch.cat(assist_stream, dim=1)

        return self._run_default_stream_response(
            input_ids=input_ids,
            fs=fs,
            n_time_block=n_time_block,
            env_stream=env_tensor,
            assist_stream=assist_tensor,
            pad_stream=pad_stream,
        )

    def _run_default_stream_response(
        self,
        *,
        input_ids: torch.Tensor,
        fs: int,
        n_time_block: int,
        env_stream: torch.Tensor,
        assist_stream: torch.Tensor,
        pad_stream: torch.Tensor | None,
    ) -> str:
        if self.data_args.lhotse_fbank:
            env_fbank_feature = self.fbank.extract(env_stream.squeeze(), sampling_rate=fs).unsqueeze(0).to(self.device)
            assist_fbank_feature = self.fbank.extract(assist_stream.squeeze(), sampling_rate=fs).unsqueeze(0).to(self.device)
        else:
            env_fbank_feature = self.fbank([env_stream.squeeze()])[0].unsqueeze(0).to(self.device)
            assist_fbank_feature = self.fbank([assist_stream.squeeze()])[0].unsqueeze(0).to(self.device)
        env_fbank_feature_len = torch.tensor([env_fbank_feature.size(1)], dtype=torch.long, device=self.device)
        assist_fbank_feature_len = torch.tensor([assist_fbank_feature.size(1)], dtype=torch.long, device=self.device)

        think_ids = self._tokenize(self.tokenizer.think_token)
        n_step = 1
        cur_text_idx = 1
        speaking = False
        end_text = False
        output_ids = []

        def generate_next_token(cur_input_ids):
            return self.model.generate(
                input_ids=cur_input_ids.unsqueeze(0).to(self.device),
                attention_mask=torch.ones_like(cur_input_ids).unsqueeze(0).to(self.device),
                env_fbank_feature=env_fbank_feature,
                env_fbank_feature_len=env_fbank_feature_len,
                assist_fbank_feature=assist_fbank_feature,
                assist_fbank_feature_len=assist_fbank_feature_len,
                n_speech_tokens=[(cur_input_ids == self.tokenizer.audio_pad_token_id).sum().item()],
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        while True:
            synthesize = False
            outputs = generate_next_token(input_ids)
            token_id = int(outputs.sequences[0, -1].item())

            if (
                (not speaking and self.eval_args.oracle_start and n_step == n_time_block)
                or (not speaking and not self.eval_args.oracle_start and token_id == 128006)
                or (speaking and token_id == 128006)
            ):
                if not speaking:
                    speaking = True
                    _ = (
                        torch.zeros((1, int(fs * self.data_args.delta * n_step)))
                        if pad_stream is None
                        else torch.cat((pad_stream, torch.zeros((1, int(fs * self.data_args.delta * n_step)))), dim=1)
                    )
                    input_ids = torch.cat((input_ids, self._tokenize("<|start_header_id|>assistant<|end_header_id|>\n\n")))
                    for idx in range(self.data_args.num_text_tokens):
                        outputs = generate_next_token(input_ids)
                        output_ids.append(outputs.sequences[:, -1:])
                        if idx < self.data_args.num_text_tokens - 1:
                            input_ids = torch.cat((input_ids, outputs.sequences[0, -1:].cpu()))
                    synthesize = True
                else:
                    break
            elif token_id == 128009:
                output_ids.append(outputs.sequences[:, -1:])
                synthesize = True
                end_text = True
                break
            else:
                if speaking:
                    if end_text:
                        outputs.sequences[:, -1:] = think_ids
                        synthesize = True
                    else:
                        output_ids.append(outputs.sequences[:, -1:])
                        if cur_text_idx == self.data_args.num_text_tokens:
                            synthesize = True
                        else:
                            input_ids = torch.cat((input_ids, outputs.sequences[0, -1:].cpu()))
                            cur_text_idx += 1
                else:
                    n_step += 1
                    input_ids = torch.cat(
                        (
                            input_ids,
                            self._tokenize(self.tokenizer.audio_pad_token * self.num_audio_embs),
                            think_ids,
                        )
                    )

            if synthesize:
                n_step += 1
                input_ids = torch.cat(
                    (
                        input_ids,
                        self._tokenize(self.tokenizer.audio_pad_token * self.num_audio_embs),
                        outputs.sequences[0, -1:].cpu(),
                    )
                )
                cur_text_idx = 1

            if n_step > 600:
                break

        if not output_ids:
            return ""
        out_text = self.tokenizer.batch_decode(torch.cat(output_ids, dim=1), skip_special_tokens=True)[0].strip()
        return " ".join(out_text.split())

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            padding=False,
            return_token_type_ids=False,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

    @staticmethod
    def _load_audio(path: str) -> tuple[torch.Tensor, int]:
        audio, sampling_rate = torchaudio.load(path)
        if sampling_rate != 16000:
            audio = torchaudio.transforms.Resample(sampling_rate, 16000)(audio)
            sampling_rate = 16000
        return audio[:1].contiguous(), sampling_rate
