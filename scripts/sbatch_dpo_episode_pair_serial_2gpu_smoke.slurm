#!/usr/bin/env bash
#SBATCH --account=lp_nelf
#SBATCH --job-name=dpo-episode-pair-serial
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --output=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.out
#SBATCH --error=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo"
QWEN_ENV="${ROOT_DIR}/envs/qwen-vllm-py312"
FULL_DUPLEX_PYTHON="${FULL_DUPLEX_PYTHON:-/data/leuven/362/vsc36212/miniconda3/envs/salmonn/bin/python}"
TTS_PYTHON="${TTS_PYTHON:-/scratch/leuven/362/vsc36212/playground/tts_eval/envs/cosyvoice/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/dpo_episode_pair_serial_smoke/${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME:-/scratch/leuven/362/vsc36212/HuggingFace}"

mkdir -p "${ROOT_DIR}/logs" "${OUT_DIR}"
cd "${ROOT_DIR}"

module purge
module load cluster/wice/gpu_h100
module load CUDA/12.1.1

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/compat:${PYTHONPATH:-}"
export NUM_TOPICS="${NUM_TOPICS:-5}"
export NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
export NUM_TURNS="${NUM_TURNS:-5}"
export TOPIC_OFFSET="${TOPIC_OFFSET:-0}"
export QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
export QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-8192}"
export QWEN_CUDA_VISIBLE_DEVICES="${QWEN_CUDA_VISIBLE_DEVICES:-0,1}"
export TTS_CUDA_VISIBLE_DEVICES="${TTS_CUDA_VISIBLE_DEVICES:-0}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "FULL_DUPLEX_PYTHON=${FULL_DUPLEX_PYTHON}"
echo "TTS_PYTHON=${TTS_PYTHON}"
echo "HF_HOME=${HF_HOME}"
echo "OUT_DIR=${OUT_DIR}"
echo "NUM_TOPICS=${NUM_TOPICS}"
echo "NUM_ROLLOUTS=${NUM_ROLLOUTS}"
echo "NUM_TURNS=${NUM_TURNS}"
echo "TOPIC_OFFSET=${TOPIC_OFFSET}"
echo "QWEN_CUDA_VISIBLE_DEVICES=${QWEN_CUDA_VISIBLE_DEVICES}"
echo "TTS_CUDA_VISIBLE_DEVICES=${TTS_CUDA_VISIBLE_DEVICES}"

PYTHONPATH="${ROOT_DIR}/compat:/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dev:/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/models:/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/third_party/Matcha-TTS:${PYTHONPATH:-}" \
"${FULL_DUPLEX_PYTHON}" - <<'PY'
import importlib
for name in ("torch", "torchaudio", "transformers", "peft", "kaldifeat", "kaldifst", "lhotse", "safetensors", "cosyvoice"):
    importlib.import_module(name)
print("full_duplex_imports_ok")
PY

"${QWEN_ENV}/bin/python" scripts/orchestrate_episode_pair_serial_smoke.py \
  --limit "${NUM_TOPICS}" \
  --offset "${TOPIC_OFFSET}" \
  --rollouts "${NUM_ROLLOUTS}" \
  --turns "${NUM_TURNS}" \
  --artifact-dir "${OUT_DIR}" \
  --output-jsonl "${OUT_DIR}/episode_pairs.jsonl" \
  --full-duplex-python "${FULL_DUPLEX_PYTHON}" \
  --tts-python "${TTS_PYTHON}"

echo "Smoke output JSONL: ${OUT_DIR}/episode_pairs.jsonl"
wc -l "${OUT_DIR}/episode_pairs.jsonl"
du -sh "${OUT_DIR}"
