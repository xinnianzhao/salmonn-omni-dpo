#!/usr/bin/env bash
#SBATCH --account=lp_nelf
#SBATCH --job-name=tts-cosyvoice2-smoke
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.out
#SBATCH --error=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo"
COSY_ENV="/scratch/leuven/362/vsc36212/playground/tts_eval/envs/cosyvoice"
PROMPT_WAV="/scratch/leuven/362/vsc36212/projects/salmonn-omni/playground/data/cosy_prompt.wav"
OUT_DIR="${ROOT_DIR}/artifacts/tts_smoke/${SLURM_JOB_ID}"
PORT="${PORT:-8021}"

mkdir -p "${ROOT_DIR}/logs" "${OUT_DIR}"
cd "${ROOT_DIR}"

module purge
module load cluster/wice/gpu_h100
module load CUDA/12.1.1

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo}"
export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-/lustre1/project/stg_00197/projects/salmonn-omni/playground/tts/CosyVoice2-0.5B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache/tts_smoke/${SLURM_JOB_ID}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${XDG_CACHE_HOME}/numba}"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "COSY_ENV=${COSY_ENV}"
echo "COSYVOICE_REPO_DIR=${COSYVOICE_REPO_DIR}"
echo "COSYVOICE_MODEL_DIR=${COSYVOICE_MODEL_DIR}"
echo "PROMPT_WAV=${PROMPT_WAV}"
echo "OUT_DIR=${OUT_DIR}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"

"${COSY_ENV}/bin/python" - <<'PY'
import sys
import torch
import torchaudio
print("python", sys.version)
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

"${COSY_ENV}/bin/python" -m uvicorn tts_interface.server:app --host 127.0.0.1 --port "${PORT}" &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl -fsS "http://127.0.0.1:${PORT}/health"
echo

"${COSY_ENV}/bin/python" scripts/test_tts_service.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --output-dir "${OUT_DIR}" \
  --mode cross_lingual \
  --prompt-audio-path "${PROMPT_WAV}" \
  --request-id "cosyvoice2_smoke" \
  --text "<|en|>I want to plan a weekend trip to Kyoto."

ls -lh "${OUT_DIR}"
python3 - <<PY
from pathlib import Path
path = Path("${OUT_DIR}") / "cosyvoice2_smoke.wav"
print("generated_wav", path, "exists", path.exists(), "bytes", path.stat().st_size if path.exists() else 0)
PY
