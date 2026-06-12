#!/usr/bin/env bash
#SBATCH --account=lp_nelf
#SBATCH --job-name=fd-api-smoke
#SBATCH --time=03:00:00
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
SALMONN_PY="${SALMONN_PY:-/data/leuven/362/vsc36212/miniconda3/envs/salmonn/bin/python}"
PORT="${PORT:-8031}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/full_duplex_service_smoke/${SLURM_JOB_ID}}"
export HF_HOME="${HF_HOME:-/scratch/leuven/362/vsc36212/HuggingFace}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/compat:${PYTHONPATH:-}"

mkdir -p "${ROOT_DIR}/logs" "${OUT_DIR}"
cd "${ROOT_DIR}"

module purge
module load cluster/wice/gpu_h100
module load CUDA/12.1.1

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SALMONN_PY=${SALMONN_PY}"
echo "HF_HOME=${HF_HOME}"
echo "PORT=${PORT}"
echo "OUT_DIR=${OUT_DIR}"

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting full-duplex in-memory API"
(
  export CUDA_VISIBLE_DEVICES=0
  export ROOT_DIR="${ROOT_DIR}"
  export FULL_DUPLEX_SERVER_PYTHON="${SALMONN_PY}"
  export FULL_DUPLEX_DEVICE_ID=0
  export FULL_DUPLEX_BACKEND=in_memory
  export FULL_DUPLEX_WORK_DIR="${OUT_DIR}/service_runs"
  export PORT="${PORT}"
  bash "${ROOT_DIR}/scripts/start_full_duplex_service.sh"
) >"${OUT_DIR}/full_duplex_service.log" 2>&1 &

elapsed=0
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  sleep 5
  elapsed=$((elapsed + 5))
  if (( elapsed >= 180 )); then
    echo "ERROR: timed out waiting for full-duplex API" >&2
    tail -n 120 "${OUT_DIR}/full_duplex_service.log" || true
    exit 3
  fi
done
echo "Full-duplex API ready"
curl -fsS "http://127.0.0.1:${PORT}/health"
echo

"${SALMONN_PY}" scripts/smoke_full_duplex_service.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --user-audio "${ROOT_DIR}/asset/user_prompt.wav" \
  --output-json "${OUT_DIR}/result.json" \
  --session-id "api_smoke_${SLURM_JOB_ID}"

echo "Result JSON: ${OUT_DIR}/result.json"
cat "${OUT_DIR}/result.json"
