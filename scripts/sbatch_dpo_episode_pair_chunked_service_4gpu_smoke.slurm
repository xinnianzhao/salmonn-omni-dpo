#!/usr/bin/env bash
#SBATCH --account=lp_nelf
#SBATCH --job-name=dpo-episode-api
#SBATCH --time=12:00:00
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
TTS_ENV="${TTS_ENV:-/scratch/leuven/362/vsc36212/playground/tts_eval/envs/cosyvoice}"
FULL_DUPLEX_PYTHON="${FULL_DUPLEX_PYTHON:-/data/leuven/362/vsc36212/miniconda3/envs/salmonn/bin/python}"
export HF_HOME="${HF_HOME:-/scratch/leuven/362/vsc36212/HuggingFace}"

TOPIC_FILE="${TOPIC_FILE:-/lustre1/scratch/362/vsc36212/dataset/dpo/duplex_data/conv_natural_trivia_220k.json}"
TOPIC_OFFSET="${TOPIC_OFFSET:-0}"
NUM_TOPICS="${NUM_TOPICS:-5}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-2}"
NUM_TURNS="${NUM_TURNS:-5}"
QWEN_PORT="${QWEN_PORT:-8001}"
TTS_PORT="${TTS_PORT:-8021}"
FULL_DUPLEX_PORT="${FULL_DUPLEX_PORT:-8031}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/dpo_episode_pair_chunked_service_smoke/${SLURM_JOB_ID}}"

mkdir -p "${ROOT_DIR}/logs" "${OUT_DIR}"
cd "${ROOT_DIR}"

module purge
module load cluster/wice/gpu_h100
module load CUDA/12.1.1

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "FULL_DUPLEX_PYTHON=${FULL_DUPLEX_PYTHON}"
echo "TTS_ENV=${TTS_ENV}"
echo "HF_HOME=${HF_HOME}"
echo "OUT_DIR=${OUT_DIR}"
echo "TOPIC_FILE=${TOPIC_FILE}"
echo "TOPIC_OFFSET=${TOPIC_OFFSET}"
echo "NUM_TOPICS=${NUM_TOPICS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "NUM_ROLLOUTS=${NUM_ROLLOUTS}"
echo "NUM_TURNS=${NUM_TURNS}"

cleanup() {
  jobs -pr | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting Qwen vLLM on GPUs 0,1 port ${QWEN_PORT}"
(
  export CUDA_VISIBLE_DEVICES=0,1
  export PORT="${QWEN_PORT}"
  export TENSOR_PARALLEL_SIZE=2
  export GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
  export MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
  export MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-8192}"
  bash "${ROOT_DIR}/scripts/start_qwen3_32b_vllm.sh"
) >"${OUT_DIR}/qwen_server.log" 2>&1 &

echo "Starting TTS service on GPU 2 port ${TTS_PORT}"
(
  export CUDA_VISIBLE_DEVICES=2
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
  export COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo}"
  export COSYVOICE_MODEL_DIR="${COSYVOICE_MODEL_DIR:-/lustre1/project/stg_00197/projects/salmonn-omni/playground/tts/CosyVoice2-0.5B}"
  export XDG_CACHE_HOME="${ROOT_DIR}/.cache/dpo_episode_pair_chunked_service_smoke/${SLURM_JOB_ID}/tts"
  export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
  export NUMBA_CACHE_DIR="${XDG_CACHE_HOME}/numba"
  mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
  "${TTS_ENV}/bin/python" -m uvicorn tts_interface.server:app --host 127.0.0.1 --port "${TTS_PORT}"
) >"${OUT_DIR}/tts_server.log" 2>&1 &

echo "Starting full-duplex API on GPU 3 port ${FULL_DUPLEX_PORT}"
(
  export CUDA_VISIBLE_DEVICES=3
  export ROOT_DIR="${ROOT_DIR}"
  export QWEN_ENV="${QWEN_ENV}"
  export PORT="${FULL_DUPLEX_PORT}"
  export FULL_DUPLEX_DEVICE_ID=0
  export FULL_DUPLEX_PYTHON="${FULL_DUPLEX_PYTHON}"
  export FULL_DUPLEX_WORK_DIR="${OUT_DIR}/full_duplex_service"
  bash "${ROOT_DIR}/scripts/start_full_duplex_service.sh"
) >"${OUT_DIR}/full_duplex_service.log" 2>&1 &

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
      tail -n 80 "${OUT_DIR}/qwen_server.log" || true
      tail -n 80 "${OUT_DIR}/tts_server.log" || true
      tail -n 80 "${OUT_DIR}/full_duplex_service.log" || true
      exit 3
    fi
  done
  echo "${name} ready at ${url}"
}

wait_for_http "Qwen" "http://127.0.0.1:${QWEN_PORT}/v1/models" 1800
wait_for_http "TTS" "http://127.0.0.1:${TTS_PORT}/health" 900
wait_for_http "Full-duplex" "http://127.0.0.1:${FULL_DUPLEX_PORT}/health" 120

echo "Running chunked orchestrator against full-duplex API"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
"${QWEN_ENV}/bin/python" scripts/orchestrate_full_duplex_responses_chunked.py \
  --topic-file "${TOPIC_FILE}" \
  --offset "${TOPIC_OFFSET}" \
  --limit "${NUM_TOPICS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --rollouts "${NUM_ROLLOUTS}" \
  --turns "${NUM_TURNS}" \
  --artifact-dir "${OUT_DIR}/artifacts" \
  --output-jsonl "${OUT_DIR}/episode_pairs.jsonl" \
  --qwen-base-url "http://127.0.0.1:${QWEN_PORT}/v1" \
  --tts-base-url "http://127.0.0.1:${TTS_PORT}" \
  --full-duplex-base-url "http://127.0.0.1:${FULL_DUPLEX_PORT}" \
  --tts-mode cross_lingual \
  --tts-user-prompt-audio "${ROOT_DIR}/asset/user_prompt.wav" \
  --tts-assistant-prompt-audio "${ROOT_DIR}/asset/assis_prompt.wav"

echo "Smoke output JSONL: ${OUT_DIR}/episode_pairs.jsonl"
wc -l "${OUT_DIR}/episode_pairs.jsonl"
du -sh "${OUT_DIR}"

