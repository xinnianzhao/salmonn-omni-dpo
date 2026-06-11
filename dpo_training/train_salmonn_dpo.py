#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dpo_training.pair_dataset import SalmonnDPOPairDataset, load_dpo_pairs
from dpo_training.salmonn_collator import DPOPairCollator


@dataclass
class DPOArguments:
    salmonn_repo: str = field(default="../salmonn-omni-dev")
    last_turn_dpo_path: Optional[str] = field(default=None)
    episode_dpo_path: Optional[str] = field(default=None)
    use_episode_pair: bool = field(default=False)
    episode_sample_ratio: float = field(default=0.25)
    episode_weight: float = field(default=0.5)
    beta: float = field(default=0.05)
    lambda_sft: float = field(default=0.1)
    average_log_prob: bool = field(default=False)
    train_lora_only: bool = field(default=True)


def _import_salmonn(salmonn_repo: str):
    salmonn_path = Path(salmonn_repo).resolve()
    if not salmonn_path.exists():
        raise FileNotFoundError(f"SALMONN repo not found: {salmonn_path}")
    sys.path.insert(0, str(salmonn_path))
    from train_hf import DataArguments, ModelArguments, TrainingArguments
    from utils import load_model
    from datasets import SALMONN_omni_Dataset

    return ModelArguments, DataArguments, TrainingArguments, load_model, SALMONN_omni_Dataset


def _freeze_reference_model(model: torch.nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def _train_lora_only(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(("lora_A" in name) or ("lora_B" in name))


def _count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device) for v in batch)
    return batch


def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor, average: bool) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
    shift_weights = loss_weights[:, 1:].contiguous().to(shift_logits.device)
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    weighted = token_logps * shift_weights * mask
    summed = weighted.sum(dim=-1)
    if average:
        denom = (shift_weights * mask).sum(dim=-1).clamp_min(1.0)
        return summed / denom
    return summed


class SalmonnDPOTrainer(Trainer):
    def __init__(self, *args, ref_model, dpo_args: DPOArguments, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model.to(self.args.device)
        self.dpo_args = dpo_args

    def _forward_logps(self, model, batch):
        outputs = model(**batch)
        logps = _sequence_logps(
            outputs.logits,
            batch["labels"],
            batch["loss_weights"],
            average=self.dpo_args.average_log_prob,
        )
        return logps, outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_device = next(model.parameters()).device
        pair_weight = inputs.pop("pair_weight").to(model_device)
        inputs.pop("pair_type", None)

        chosen = _move_to_device(inputs["chosen"], model_device)
        rejected = _move_to_device(inputs["rejected"], model_device)

        pi_chosen_logps, chosen_outputs = self._forward_logps(model, chosen)
        pi_rejected_logps, _ = self._forward_logps(model, rejected)

        with torch.no_grad():
            ref_chosen = _move_to_device(inputs["chosen"], next(self.ref_model.parameters()).device)
            ref_rejected = _move_to_device(inputs["rejected"], next(self.ref_model.parameters()).device)
            ref_chosen_logps, _ = self._forward_logps(self.ref_model, ref_chosen)
            ref_rejected_logps, _ = self._forward_logps(self.ref_model, ref_rejected)

        ref_chosen_logps = ref_chosen_logps.to(pi_chosen_logps.device)
        ref_rejected_logps = ref_rejected_logps.to(pi_rejected_logps.device)

        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        dpo_logits = pi_logratios - ref_logratios
        dpo_losses = -F.logsigmoid(self.dpo_args.beta * dpo_logits)
        dpo_loss = (dpo_losses * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)

        sft_loss = chosen_outputs.loss if chosen_outputs.loss is not None else torch.zeros_like(dpo_loss)
        loss = dpo_loss + self.dpo_args.lambda_sft * sft_loss
        preference_accuracy = (dpo_logits > 0).float().mean()
        chosen_reward = self.dpo_args.beta * (pi_chosen_logps - ref_chosen_logps)
        rejected_reward = self.dpo_args.beta * (pi_rejected_logps - ref_rejected_logps)
        reward_margin = chosen_reward - rejected_reward
        approx_kl = (pi_chosen_logps - ref_chosen_logps).detach().float().mean()

        self.log(
            {
                "dpo_loss": dpo_loss.detach().float().item(),
                "sft_loss": sft_loss.detach().float().item(),
                "total_loss": loss.detach().float().item(),
                "preference_accuracy": preference_accuracy.detach().float().item(),
                "preference_margin": dpo_logits.detach().float().mean().item(),
                "chosen_logp": pi_chosen_logps.detach().float().mean().item(),
                "rejected_logp": pi_rejected_logps.detach().float().mean().item(),
                "chosen_reward": chosen_reward.detach().float().mean().item(),
                "rejected_reward": rejected_reward.detach().float().mean().item(),
                "reward_margin": reward_margin.detach().float().mean().item(),
                "approx_kl_chosen": approx_kl.item(),
            }
        )
        return (loss, {"chosen_outputs": chosen_outputs}) if return_outputs else loss


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--salmonn_repo", default="../salmonn-omni-dev")
    pre_args, _ = pre_parser.parse_known_args()

    ModelArguments, DataArguments, TrainingArguments, load_model, SALMONN_omni_Dataset = _import_salmonn(
        pre_args.salmonn_repo
    )
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, DPOArguments))
    model_args, data_args, training_args, dpo_args = parser.parse_args_into_dataclasses()

    data_args.lhotse_fbank = model_args.speech_encoder_type in ("zipformer2", "transformer", "transformer_v2")
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    os.makedirs(training_args.output_dir, exist_ok=True)
    config_path = Path(training_args.output_dir) / "dpo_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_args": asdict(model_args),
                "data_args": asdict(data_args),
                "training_args": training_args.to_dict(),
                "dpo_args": asdict(dpo_args),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    tokenizer, policy_model = load_model(model_args, data_args)
    _, ref_model = load_model(model_args, data_args)
    _freeze_reference_model(ref_model)

    if dpo_args.train_lora_only:
        _train_lora_only(policy_model)
    trainable, total = _count_trainable(policy_model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / max(total, 1):.4%})")

    pairs = load_dpo_pairs(
        dpo_args.last_turn_dpo_path,
        dpo_args.episode_dpo_path,
        dpo_args.use_episode_pair,
        dpo_args.episode_sample_ratio,
        dpo_args.episode_weight,
        int(training_args.seed),
    )
    print(f"Loaded {len(pairs):,} DPO pairs")
    train_dataset = SalmonnDPOPairDataset(pairs, SALMONN_omni_Dataset, model_args, data_args, tokenizer)

    trainer = SalmonnDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        dpo_args=dpo_args,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DPOPairCollator(),
    )
    trainer.train(resume_from_checkpoint=bool(list(Path(training_args.output_dir).glob("checkpoint-*"))))
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
