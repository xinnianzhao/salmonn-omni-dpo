from __future__ import annotations

from typing import Any

import torch


class SalmonnOmniCollator:
    """Collate already-tokenized SALMONN-Omni samples.

    This mirrors the collator in salmonn-omni-dev/train_hf.py so DPO training
    feeds the model exactly the same tensor fields as SFT training.
    """

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        max_input_len = max(s["input_len"] for s in samples)

        input_ids = torch.stack([s["input_ids"] for s in samples])[:, :max_input_len]
        attention_mask = torch.stack([s["attention_mask"] for s in samples])[:, :max_input_len]
        input_type_ids = torch.stack([s["input_type_ids"] for s in samples])[:, :max_input_len]
        labels = torch.stack([s["labels"] for s in samples])[:, :max_input_len]
        loss_weights = torch.stack([s["loss_weights"] for s in samples])[:, :max_input_len]

        env_fbank_feature_len = torch.cat([s["env_fbank_feature_lens"] for s in samples], dim=0)
        assist_fbank_feature_len = torch.cat([s["assist_fbank_feature_lens"] for s in samples], dim=0)
        env_fbank_feature = torch.cat([s["env_fbank_features"] for s in samples], dim=0)[
            :, : env_fbank_feature_len.max()
        ]
        assist_fbank_feature = torch.cat([s["assist_fbank_features"] for s in samples], dim=0)[
            :, : assist_fbank_feature_len.max()
        ]

        result: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_type_ids": input_type_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "env_fbank_feature": env_fbank_feature,
            "env_fbank_feature_len": env_fbank_feature_len,
            "assist_fbank_feature": assist_fbank_feature,
            "assist_fbank_feature_len": assist_fbank_feature_len,
            "n_speech_tokens": [tok for s in samples for tok in s["n_speech_tokens"]],
            "tts_features": [s["tts_features"] for s in samples],
            "sent_lens": [s["sent_lens"] for s in samples],
        }

        if "latent_reason_ids" in samples[0]:
            result["latent_reason_ids"] = torch.stack([s["latent_reason_ids"] for s in samples])
            result["latent_reason_len"] = torch.tensor([s["latent_reason_len"] for s in samples])
            result["latent_mask"] = torch.stack([s["latent_mask"] for s in samples])[:, :max_input_len]

        if "duplex_labels" in samples[0]:
            result["duplex_labels"] = torch.stack([s["duplex_labels"] for s in samples])[:, :max_input_len]

        if "reason_level_label" in samples[0]:
            result["reason_level_labels"] = torch.tensor(
                [s["reason_level_label"] for s in samples], dtype=torch.long
            )
            result["reason_level_positions"] = torch.tensor(
                [s["reason_level_position"] for s in samples], dtype=torch.long
            )

        return result


class DPOPairCollator:
    def __init__(self) -> None:
        self.base_collator = SalmonnOmniCollator()

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        chosen = self.base_collator([s["chosen"] for s in samples])
        rejected = self.base_collator([s["rejected"] for s in samples])
        return {
            "chosen": chosen,
            "rejected": rejected,
            "pair_weight": torch.tensor([s.get("pair_weight", 1.0) for s in samples], dtype=torch.float32),
            "pair_type": [s.get("pair_type", "") for s in samples],
        }

