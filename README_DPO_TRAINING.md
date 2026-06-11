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

## Judge/DPO Conversion

Keep the raw orchestration output for debugging:

```text
episode_pairs.jsonl
```

Then extract compact text-only files for Claude:

```bash
python3 scripts/extract_pairs_for_judge.py \
  --input-jsonl artifacts/dpo_episode_pair_smoke/60523385/episode_pairs.jsonl \
  --episode-output artifacts/judge/episode_pairs_for_judge.jsonl \
  --last-turn-output artifacts/judge/last_turn_pairs_for_judge.jsonl \
  --last-turn 5
```

The judge files contain only IDs, topic/source, text conversations, candidates,
and judge instructions. They intentionally omit audio paths, latency, commands,
tracebacks, and model configs.

After Claude returns judge results with fields like:

```json
{"id": "ep0", "pair_type": "episode", "winner": "A", "reason": "..."}
{"id": "ep0_turn5", "raw_session_id": "ep0", "pair_type": "last_turn", "target_turn": 5, "winner": "B", "reason": "..."}
```

merge them back with the raw file to make trainer-ready DPO pairs:

```bash
python3 scripts/prepare_dpo_pairs.py \
  --raw-jsonl artifacts/dpo_episode_pair_smoke/60523385/episode_pairs.jsonl \
  --judge-json artifacts/judge/claude_judged_pairs.jsonl \
  --episode-output artifacts/dpo/episode_pairs_dpo.jsonl \
  --last-turn-output artifacts/dpo/last_turn_pairs_dpo.jsonl
```

The DPO files preserve SALMONN `conv_v2` items under `chosen` and `rejected`,
including `user_path` and `assistant_path`, so they can be passed directly to
`dpo_training/train_salmonn_dpo.py`.

## W&B Tracking

The Slurm launcher logs to W&B by default:

```bash
WANDB_PROJECT=salmonn-omni-dpo \
WANDB_RUN_NAME=dpo-last-turn-v1 \
LAST_TURN_DPO_PATH=/path/to/last_turn_pairs.jsonl \
sbatch scripts/sbatch_salmonn_dpo.slurm
```

To disable network logging while testing:

```bash
WANDB_MODE=offline REPORT_TO=wandb ...
```

or:

```bash
REPORT_TO=none ...
```

The trainer logs `dpo_loss`, `sft_loss`, `total_loss`,
`preference_accuracy`, `preference_margin`, chosen/rejected log-probs,
implicit rewards, reward margin, and approximate KL on chosen responses.
