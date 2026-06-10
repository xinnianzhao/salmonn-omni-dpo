#!/usr/bin/env bash
#SBATCH --account=lp_nelf
#SBATCH --job-name=qwen3-32b-vllm-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --output=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-a100-%j.out
#SBATCH --error=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-a100-%j.err

set -euo pipefail

mkdir -p /lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs

export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export PORT="${PORT:-8001}"

bash /lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/scripts/start_qwen3_32b_vllm.sh
