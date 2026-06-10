"""Qwen3 user-simulator interface.

Drives Qwen3-32B (served by vLLM) to role-play the *user* side of a spoken
dialogue, producing one simulated user turn at a time for the DPO data pipeline.

- :class:`QwenUserSimulator` — the OpenAI-compatible client.
- :func:`clean_user_utterance` — strips model artifacts from a raw response.
"""

from .client import QwenUserSimulator, clean_user_utterance

__all__ = ["QwenUserSimulator", "clean_user_utterance"]
