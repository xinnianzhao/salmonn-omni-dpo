#!/usr/bin/env bash
#SBATCH --account=lp_nelf
#SBATCH --job-name=full-duplex-smoke
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.out
#SBATCH --error=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo"
FULL_DUPLEX_PYTHON="${FULL_DUPLEX_PYTHON:-/data/leuven/362/vsc36212/miniconda3/envs/salmonn/bin/python}"
FULL_DUPLEX_REPO="${FULL_DUPLEX_REPO:-/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dev}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/lustre1/project/stg_00197/projects/salmonn-omni-dpo/models/ase_er0-44k}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/full_duplex_only_smoke/${SLURM_JOB_ID}}"
USER_AUDIO="${USER_AUDIO:-${ROOT_DIR}/asset/user_prompt.wav}"
export HF_HOME="${HF_HOME:-/scratch/leuven/362/vsc36212/HuggingFace}"

mkdir -p "${ROOT_DIR}/logs" "${OUT_DIR}"
cd "${ROOT_DIR}"

module purge
module load cluster/wice/gpu_h100
module load CUDA/12.1.1

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "FULL_DUPLEX_PYTHON=${FULL_DUPLEX_PYTHON}"
echo "FULL_DUPLEX_REPO=${FULL_DUPLEX_REPO}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "USER_AUDIO=${USER_AUDIO}"
echo "OUT_DIR=${OUT_DIR}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/compat:${FULL_DUPLEX_REPO}:/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/models:/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/third_party/Matcha-TTS:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export MKL_THREADING_LAYER=GNU
export TOKENIZERS_PARALLELISM=false

"${FULL_DUPLEX_PYTHON}" scripts/smoke_full_duplex_eval.py \
  --full-duplex-repo "${FULL_DUPLEX_REPO}" \
  --full-duplex-python "${FULL_DUPLEX_PYTHON}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --model-args-json "${ROOT_DIR}/configs/full_duplex_default_model_args.json" \
  --data-args-json "${ROOT_DIR}/configs/full_duplex_default_data_args.json" \
  --eval-args-json "${ROOT_DIR}/configs/full_duplex_default_eval_args.json" \
  --user-audio "${USER_AUDIO}" \
  --artifact-dir "${OUT_DIR}" \
  --output-json "${OUT_DIR}/result.json" \
  --device-id 0

echo "Result JSON: ${OUT_DIR}/result.json"
cat "${OUT_DIR}/result.json"

