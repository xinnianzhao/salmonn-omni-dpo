from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def _read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "pairs"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported DPO data shape in {path}")


def _as_salmonn_item(value: dict[str, Any], pair_type: str) -> dict[str, Any]:
    """Return one SALMONN dataset item.

    Preferred format:
      {"chosen": {"task": "conv_v2", "text": [...]}, "rejected": {...}}

    Also accepts common aliases produced by judging scripts:
      chosen_item / rejected_item, chosen_conversation / rejected_conversation.
    """

    if "task" in value and "text" in value:
        return value
    if "item" in value and isinstance(value["item"], dict):
        return _as_salmonn_item(value["item"], pair_type)
    if "conversation" in value:
        return {"task": "conv_v2", "text": value["conversation"]}
    if "turns" in value:
        return {"task": "conv_v2", "text": value["turns"]}
    raise ValueError(f"Cannot infer SALMONN item for {pair_type} pair from keys: {sorted(value)}")


def _extract_pair(row: dict[str, Any], pair_type: str, default_weight: float) -> dict[str, Any]:
    chosen = (
        row.get("chosen")
        or row.get("chosen_item")
        or row.get("chosen_conversation")
        or row.get("winner")
    )
    rejected = (
        row.get("rejected")
        or row.get("rejected_item")
        or row.get("rejected_conversation")
        or row.get("loser")
    )
    if chosen is None or rejected is None:
        raise ValueError(f"DPO pair is missing chosen/rejected fields. Keys: {sorted(row)}")
    return {
        "chosen": _as_salmonn_item(chosen, pair_type),
        "rejected": _as_salmonn_item(rejected, pair_type),
        "pair_type": row.get("pair_type", pair_type),
        "pair_weight": float(row.get("weight", row.get("pair_weight", default_weight))),
    }


def load_dpo_pairs(
    last_turn_dpo_path: str | None,
    episode_dpo_path: str | None,
    use_episode_pair: bool,
    episode_sample_ratio: float,
    episode_weight: float,
    seed: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []

    if last_turn_dpo_path:
        for row in _read_json_or_jsonl(last_turn_dpo_path):
            pairs.append(_extract_pair(row, "last_turn", 1.0))

    if use_episode_pair:
        if not episode_dpo_path:
            raise ValueError("--use_episode_pair requires --episode_dpo_path")
        episode_pairs = [_extract_pair(row, "episode", episode_weight) for row in _read_json_or_jsonl(episode_dpo_path)]
        if pairs and 0.0 < episode_sample_ratio < 1.0:
            max_episode = math.ceil(len(pairs) * episode_sample_ratio / (1.0 - episode_sample_ratio))
            rng = random.Random(seed)
            rng.shuffle(episode_pairs)
            episode_pairs = episode_pairs[:max_episode]
        pairs.extend(episode_pairs)

    if not pairs:
        raise ValueError("No DPO pairs loaded")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


class SalmonnDPOPairDataset(Dataset):
    """Pair wrapper around SALMONN_omni_Dataset.

    SALMONN_omni_Dataset expects a flat JSON file, so this class flattens each
    preference pair into [chosen0, rejected0, chosen1, rejected1, ...] and then
    returns paired tokenized samples.
    """

    def __init__(self, pairs: list[dict[str, Any]], salmonn_dataset_cls, model_args, data_args, tokenizer):
        self.pairs = pairs
        flat_items = []
        for pair in pairs:
            flat_items.append(pair["chosen"])
            flat_items.append(pair["rejected"])

        self._tmpdir = tempfile.TemporaryDirectory(prefix="salmonn_dpo_pairs_")
        self.flat_path = Path(self._tmpdir.name) / "flat_pairs.json"
        with self.flat_path.open("w", encoding="utf-8") as f:
            json.dump(flat_items, f, ensure_ascii=False)

        data_args.dataset_dir = str(self.flat_path)
        self.flat_dataset = salmonn_dataset_cls(model_args, data_args, tokenizer)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        return {
            "chosen": self.flat_dataset[2 * index],
            "rejected": self.flat_dataset[2 * index + 1],
            "pair_weight": pair["pair_weight"],
            "pair_type": pair["pair_type"],
        }

