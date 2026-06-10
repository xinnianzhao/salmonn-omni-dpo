# Full-Duplex Response Orchestrator

This layer generates raw multi-turn records for later DPO judging. It keeps the
modules as separate local services/processes:

- Qwen3-32B user simulator via vLLM OpenAI-compatible API.
- CosyVoice2 TTS via the local TTS HTTP service.
- SALMONN-Omni full-duplex inference via `../salmonn-omni-dev/evaluate_hf.py`.

## Run Shape

For each topic, the orchestrator repeats:

1. Ask Qwen for the next user utterance.
2. Synthesize that user utterance to wav.
3. Build a one-item `conv_v2` dataset using the `default` history format and
   call `evaluate_hf.py`.
4. Save the generated assistant text.
5. Synthesize the full assistant text to one wav for later conversation
   history. Previous turns contain `user_path`, `assistant`, and
   `assistant_path`; the current target turn contains `user_path`.

The raw output is JSONL. Candidate response slots are present, but currently
empty because `evaluate_hf.py` hardcodes greedy generation (`do_sample=False`).
To create true DPO candidates, expose decoding args in `evaluate_hf.py` or add
a small service wrapper that samples multiple responses from one loaded model.

The `ase_er0-44k` checkpoint config uses `template="default"`, so the active
orchestrator path uses `task="conv_v2"` rather than the earlier experimental
`short_v2` path.

## Example

Start Qwen and TTS services on the allocated node, then run:

```bash
python scripts/orchestrate_full_duplex_responses.py \
  --topic "weekend trip to Kyoto" \
  --turns 5 \
  --full-duplex-python /path/to/full-duplex/env/bin/python \
  --output-jsonl artifacts/dpo_raw/full_duplex_raw.jsonl
```

Use `--dry-run` to test Qwen+TTS plus JSON writing without loading the
full-duplex checkpoint.
