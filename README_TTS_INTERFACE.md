# TTS Interface

This is the DPO-facing TTS interface. It keeps CosyVoice implementation and model files outside this repo while exposing one stable synthesis API for the data-generation pipeline.

## Paths

- DPO interface code: `${PROJECT_DIR}/salmonn-omni-dpo/tts_interface`
- CosyVoice code default: `/lustre1/scratch/362/vsc36212/projects/salmonn-omni/salmonn-omni-dev-mine-demo`
- CosyVoice2 model default: `/lustre1/project/stg_00197/projects/salmonn-omni/playground/tts/CosyVoice2-0.5B`

## Service

Start the service from an environment that has the CosyVoice dependencies installed:

```bash
cd /lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo
PYTHONPATH=$PWD bash scripts/start_tts_service.sh
```

The service loads CosyVoice2 once and exposes:

```text
GET  /health
POST /synthesize
```

All synthesis calls use non-streaming generation:

```python
stream=False
```

## Test

CosyVoice2 zero-shot mode requires a prompt transcript and a 16 kHz prompt wav:

```bash
python scripts/test_tts_service.py \
  --output-dir /lustre1/scratch/362/vsc36212/projects/salmonn-omni-dpo/artifacts/tts_smoke \
  --mode zero_shot \
  --prompt-text "This is the transcript of the reference voice." \
  --prompt-audio-path /path/to/reference_16k.wav \
  --text "I want to plan a weekend trip to Kyoto."
```

The response contains the generated `audio_path`, `sample_rate`, and duration.
