#!/usr/bin/env bash
#SBATCH --account=lp_nelf
#SBATCH --job-name=dpo-short-v2-smoke
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.out
#SBATCH --error=/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo"
QWEN_ENV="${ROOT_DIR}/envs/qwen-vllm-py312"
TTS_ENV="/scratch/leuven/362/vsc36212/playground/tts_eval/envs/cosyvoice"

# Must point to a Python environment that can run ../salmonn-omni-dev/evaluate_hf.py
# with torch, transformers, peft, torchaudio, kaldifeat, lhotse, safetensors, etc.
FULL_DUPLEX_PYTHON="${FULL_DUPLEX_PYTHON:-}"

TOPIC_FILE="${TOPIC_FILE:-/lustre1/scratch/362/vsc36212/dataset/dpo/duplex_data/conv_natural_trivia_220k.json}"
TOPIC_OFFSET="${TOPIC_OFFSET:-0}"
NUM_CONVERSATIONS="${NUM_CONVERSATIONS:-5}"
NUM_TURNS="${NUM_TURNS:-5}"
QWEN_PORT="${QWEN_PORT:-8001}"
TTS_PORT="${TTS_PORT:-8021}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/dpo_smoke/${SLURM_JOB_ID}}"

mkdir -p "${ROOT_DIR}/logs" "${OUT_DIR}"
cd "${ROOT_DIR}"

if [[ -z "${FULL_DUPLEX_PYTHON}" ]]; then
  echo "ERROR: set FULL_DUPLEX_PYTHON to the SALMONN-Omni evaluation environment python." >&2
  echo "Example: FULL_DUPLEX_PYTHON=/path/to/env/bin/python sbatch ${BASH_SOURCE[0]}" >&2
  exit 2
fi
if [[ ! -x "${FULL_DUPLEX_PYTHON}" ]]; then
  echo "ERROR: FULL_DUPLEX_PYTHON is not executable: ${FULL_DUPLEX_PYTHON}" >&2
  exit 2
fi

module purge
module load cluster/wice/gpu_h100
module load CUDA/12.1.1

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "FULL_DUPLEX_PYTHON=${FULL_DUPLEX_PYTHON}"
echo "OUT_DIR=${OUT_DIR}"
echo "TOPIC_FILE=${TOPIC_FILE}"
echo "TOPIC_OFFSET=${TOPIC_OFFSET}"
echo "NUM_CONVERSATIONS=${NUM_CONVERSATIONS}"
echo "NUM_TURNS=${NUM_TURNS}"

echo "Checking full-duplex Python imports"
(
  export PYTHONPATH="${ROOT_DIR}/compat:/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dev:/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/models:/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo/third_party/Matcha-TTS:${PYTHONPATH:-}"
  "${FULL_DUPLEX_PYTHON}" - <<'PY'
import importlib
for name in ("torch", "torchaudio", "transformers", "peft", "kaldifeat", "lhotse", "safetensors", "cosyvoice"):
    importlib.import_module(name)
print("full_duplex_imports_ok")
PY
)

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting Qwen vLLM on GPUs 0,1 port ${QWEN_PORT}"
(
  export CUDA_VISIBLE_DEVICES=0,1
  export PORT="${QWEN_PORT}"
  export TENSOR_PARALLEL_SIZE=2
  export MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
  export MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-8192}"
  bash "${ROOT_DIR}/scripts/start_qwen3_32b_vllm.sh"
) >"${OUT_DIR}/qwen_server.log" 2>&1 &
QWEN_PID=$!

echo "Starting TTS service on GPU 2 port ${TTS_PORT}"
(
  export CUDA_VISIBLE_DEVICES=2
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
  export COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo}"
  export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-/lustre1/project/stg_00197/projects/salmonn-omni/playground/tts/CosyVoice2-0.5B}"
  export XDG_CACHE_HOME="${ROOT_DIR}/.cache/dpo_smoke/${SLURM_JOB_ID}/tts"
  export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
  export NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
  mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
  "${TTS_ENV}/bin/python" -m uvicorn tts_interface.server:app --host 127.0.0.1 --port "${TTS_PORT}"
) >"${OUT_DIR}/tts_server.log" 2>&1 &
TTS_PID=$!

wait_for_http() {
  local name="$1"
  local url="$2"
  local max_wait="$3"
  local elapsed=0
  until curl -fsS "${url}" >/dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed >= max_wait )); then
      echo "ERROR: timed out waiting for ${name} at ${url}" >&2
      echo "Qwen log: ${OUT_DIR}/qwen_server.log" >&2
      echo "TTS log: ${OUT_DIR}/tts_server.log" >&2
      exit 3
    fi
  done
  echo "${name} ready at ${url}"
}

wait_for_http "Qwen" "http://127.0.0.1:${QWEN_PORT}/v1/models" 1800
wait_for_http "TTS" "http://127.0.0.1:${TTS_PORT}/health" 900

echo "Running orchestrator on GPU 3"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
"${QWEN_ENV}/bin/python" scripts/orchestrate_full_duplex_responses.py \
  --topic-file "${TOPIC_FILE}" \
  --offset "${TOPIC_OFFSET}" \
  --limit "${NUM_CONVERSATIONS}" \
  --turns "${NUM_TURNS}" \
  --artifact-dir "${OUT_DIR}/artifacts" \
  --output-jsonl "${OUT_DIR}/full_duplex_raw.jsonl" \
  --qwen-base-url "http://127.0.0.1:${QWEN_PORT}/v1" \
  --tts-base-url "http://127.0.0.1:${TTS_PORT}" \
  --tts-mode cross_lingual \
  --full-duplex-repo "/lustre1/scratch/362/vsc36212/projects/salmonn-omni-dev" \
  --full-duplex-python "${FULL_DUPLEX_PYTHON}" \
  --full-duplex-model-args-json "${ROOT_DIR}/configs/full_duplex_short_v2_model_args.json" \
  --full-duplex-data-args-json "${ROOT_DIR}/configs/full_duplex_short_v2_data_args.json" \
  --full-duplex-eval-args-json "${ROOT_DIR}/configs/full_duplex_short_v2_eval_args.json"

echo "Smoke output JSONL: ${OUT_DIR}/full_duplex_raw.jsonl"
wc -l "${OUT_DIR}/full_duplex_raw.jsonl"
du -sh "${OUT_DIR}"
