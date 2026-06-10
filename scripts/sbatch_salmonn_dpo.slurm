#!/bin/bash
#SBATCH --job-name=salmonn-dpo
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/salmonn-dpo-%j.out
#SBATCH --error=logs/salmonn-dpo-%j.err

set -euo pipefail

PROJECT_DIR=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo
SALMONN_REPO=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dev
SALMONN_PY=/data/leuven/362/vsc36212/miniconda3/envs/salmonn/bin/python
CKPT_DIR=/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k
BASE_MODEL=/mnt/bn/audio-visual-llm-data2/yuwenyi/ckpt/Meta-Llama-3-8B-Instruct/

: "${LAST_TURN_DPO_PATH:?Set LAST_TURN_DPO_PATH to judged last-turn pair json/jsonl}"
: "${OUTPUT_DIR:=$PROJECT_DIR/artifacts/salmonn_dpo/${SLURM_JOB_ID}}"
: "${EPISODE_DPO_PATH:=}"
: "${USE_EPISODE_PAIR:=false}"
: "${WARMUP_STEPS:=200}"
: "${LAMBDA_SFT:=0.1}"
: "${EPISODE_SAMPLE_RATIO:=0.25}"
: "${EPISODE_WEIGHT:=0.5}"
: "${BETA:=0.05}"
: "${LR:=1e-5}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=16}"
: "${MAX_STEPS:=-1}"
: "${NUM_EPOCHS:=1}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

export PYTHONPATH="$PROJECT_DIR:$SALMONN_REPO:$PROJECT_DIR/compat:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export MKL_THREADING_LAYER=GNU
export NCCL_DEBUG=ERROR
export TOKENIZERS_PARALLELISM=false

"$SALMONN_PY" -m torch.distributed.run --standalone --nproc_per_node=4 dpo_training/train_salmonn_dpo.py \
  --salmonn_repo "$SALMONN_REPO" \
  --last_turn_dpo_path "$LAST_TURN_DPO_PATH" \
  --episode_dpo_path "$EPISODE_DPO_PATH" \
  --use_episode_pair "$USE_EPISODE_PAIR" \
  --episode_sample_ratio "$EPISODE_SAMPLE_RATIO" \
  --episode_weight "$EPISODE_WEIGHT" \
  --beta "$BETA" \
  --lambda_sft "$LAMBDA_SFT" \
  --train_lora_only true \
  --model_name_or_path "$BASE_MODEL" \
  --speech_encoder_type transformer_v2 \
  --speech_enc_llm_proj_setting 4 4096 \
  --ckpt_path "$CKPT_DIR" \
  --speech_synthesizer_type none \
  --add_assist_stream true \
  --num_text_tokens 1 \
  --dataset_dir "$LAST_TURN_DPO_PATH" \
  --delta 0.08 \
  --simple_mode true \
  --simple_mode_echo_ratio 0.0 \
  --same_prompt true \
  --add_end_of_speech true \
  --template default \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_train_epochs "$NUM_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --learning_rate "$LR" \
  --warmup_steps "$WARMUP_STEPS" \
  --bf16 true \
  --logging_steps 5 \
  --save_steps 500 \
  --save_total_limit 3 \
  --remove_unused_columns false \
  --report_to none
