# Full-Duplex API Wrapper

This service lives in `salmonn-omni-dpo` and leaves the upstream
`salmonn-omni-dev` checkout unchanged.

Default backend:

```text
HTTP request -> in-memory conv_v2 generator -> loaded SALMONN model
```

The service runs in the SALMONN conda environment, loads the model on the first
generation request, then keeps it resident for later requests. The upstream
`salmonn-omni-dev` checkout is imported but not modified.

Fallback backend:

```bash
FULL_DUPLEX_BACKEND=subprocess bash scripts/start_full_duplex_service.sh
```

This keeps the same API but delegates each request/batch to `evaluate_hf.py`.

## Start

```bash
CUDA_VISIBLE_DEVICES=0 \
PORT=8031 \
FULL_DUPLEX_BACKEND=in_memory \
bash scripts/start_full_duplex_service.sh
```

Health:

```bash
curl http://127.0.0.1:8031/health
```

## API

Single turn:

```http
POST /generate_turn
```

```json
{
  "session_id": "ep0_A",
  "turn_id": 1,
  "run_id": "ep0_A_turn01",
  "conv_turns": [
    {
      "user": "What exactly is considered the Outer Banks?",
      "user_path": "/path/to/user.wav"
    }
  ]
}
```

Batch of same-turn requests:

```http
POST /generate_batch
```

```json
{
  "batch_id": "chunk0000_turn01",
  "requests": [
    {
      "session_id": "ep0_A",
      "turn_id": 1,
      "run_id": "ep0_A_turn01",
      "conv_turns": [{"user": "...", "user_path": "..."}]
    },
    {
      "session_id": "ep0_B",
      "turn_id": 1,
      "run_id": "ep0_B_turn01",
      "conv_turns": [{"user": "...", "user_path": "..."}]
    }
  ]
}
```

All requests in one batch must use the same `turn_id`, because
`evaluate_hf.py` has one global `--num_turn` argument.

## Chunked Smoke

This starts Qwen on GPUs 0-1, TTS on GPU 2, and the full-duplex API on GPU 3:

```bash
sbatch scripts/sbatch_dpo_episode_pair_chunked_service_4gpu_smoke.slurm
```

Test only the full-duplex API:

```bash
sbatch scripts/sbatch_full_duplex_service_smoke.slurm
```

This starts the service on one H100, sends one `conv_v2` request through
`/generate_turn`, and writes:

```text
artifacts/full_duplex_service_smoke/${SLURM_JOB_ID}/result.json
```

Full 4-GPU chunked smoke:

```bash
sbatch scripts/sbatch_dpo_episode_pair_chunked_service_4gpu_smoke.slurm
```

The chunked orchestrator uses the API via:

```bash
--full-duplex-base-url http://127.0.0.1:8031
```
