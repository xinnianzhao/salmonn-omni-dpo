# Qwen3-32B Local User Simulator

This setup runs Qwen3-32B locally through vLLM's OpenAI-compatible API, then optionally exposes a smaller `/generate_user` API for the DPO data pipeline.

## Paths

- Code: `${PROJECT_DIR}/salmonn-omni-dpo`
- Python env: `${PROJECT_DIR}/salmonn-omni-dpo/envs/qwen-vllm-py312`
- Model/cache data: `/{CACHE_DIR}/salmonn-omni-dpo`

## 1. Create Environment

Run on an HPC node that can access the package index:

```bash
bash scripts/setup_qwen_vllm_env.sh
```

The script uses Python 3.12, CUDA 13, and the NVIDIA GPU vLLM recipe style:

```bash
uv pip install -U vllm --torch-backend=auto
```

## 2. Start vLLM

On a GPU node:

```bash
bash scripts/start_qwen3_32b_vllm.sh
```

Useful overrides:

```bash
TENSOR_PARALLEL_SIZE=4 PORT=8001 GPU_MEMORY_UTILIZATION=0.90 bash scripts/start_qwen3_32b_vllm.sh
```

For Slurm:

```bash
sbatch scripts/sbatch_qwen3_32b_vllm.slurm
```

## 3. Test OpenAI-Compatible API

```bash
source scripts/activate_qwen_env.sh
python scripts/test_qwen_vllm_api.py
```

## 4. Optional User-Simulator API

Keep vLLM running, then start the simulator wrapper:

```bash
bash scripts/start_qwen_user_sim_api.sh
```

Call it:

```bash
curl -s http://127.0.0.1:8011/generate_user \
  -H 'Content-Type: application/json' \
  -d '{
    "topic": "planning a trip to Japan",
    "turn_id": 1,
    "history": []
  }'
```

The wrapper returns a structured object with the simulated user utterance.