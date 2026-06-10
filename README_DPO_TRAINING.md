# SALMONN-Omni DPO Training

This repo keeps DPO training outside `salmonn-omni-dev`. The policy and
reference models are loaded with the same SALMONN model/config/checkpoint code
used by evaluation; only the trainer and paired dataset wrapper are different.

## Data Format

The preferred input is JSONL or JSON with one DPO pair per row:

```json
{
  "pair_type": "last_turn",
  "chosen": {"task": "conv_v2", "text": [{"user": "...", "user_path": "...", "assistant": "...", "assistant_path": "..."}]},
  "rejected": {"task": "conv_v2", "text": [{"user": "...", "user_path": "...", "assistant": "...", "assistant_path": "..."}]},
  "weight": 1.0
}
```

Accepted aliases are `chosen_item`/`rejected_item`,
`chosen_conversation`/`rejected_conversation`, and `winner`/`loser`. If a value
contains `conversation` or `turns`, it is wrapped as a `conv_v2` SALMONN item.

## Initial Hyperparameters

- `warmup_steps=200`
- `lambda_sft=0.1`
- `use_episode_pair=false` for the first run
- `beta=0.05`
- `learning_rate=1e-5`
- effective batch size about 64 if memory allows

After the last-turn-only run is stable, enable episode pairs:

- `use_episode_pair=true`
- `episode_sample_ratio=0.25`
- `episode_weight=0.5`

This uses all last-turn pairs and samples episode pairs so they are about 25%
of the mixed dataset. Episode pairs are longer and noisier, so they should not
dominate the first DPO run.

## Submit

```bash
LAST_TURN_DPO_PATH=/path/to/last_turn_pairs.jsonl \
sbatch scripts/sbatch_salmonn_dpo.slurm
```

Mixed last-turn plus episode training:

```bash
LAST_TURN_DPO_PATH=/path/to/last_turn_pairs.jsonl \
EPISODE_DPO_PATH=/path/to/episode_pairs.jsonl \
USE_EPISODE_PAIR=true \
sbatch scripts/sbatch_salmonn_dpo.slurm
```

The trainer defaults to LoRA-only updates with a frozen SFT reference model. The
base Llama backbone and reference model remain frozen.

