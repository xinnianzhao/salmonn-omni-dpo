"""Qwen3-32B user-simulator client.

This module drives a **user simulator** for the DPO data pipeline. Given the
running conversation (topic + history), it asks Qwen3-32B for the *next user
utterance* — i.e. it role-plays the human side of a spoken dialogue so the
full-duplex SALMONN-Omni assistant can be probed over many turns without a real
person in the loop.

Transport
---------
The class talks to a vLLM server that exposes an **OpenAI-compatible** Chat
Completions API (started by ``scripts/start_qwen3_32b_vllm.sh``). We therefore
reuse the official ``openai`` SDK and only need a ``base_url`` pointing at vLLM;
the API key is unused by vLLM but the SDK requires a non-empty value.

Output contract
---------------
``generate_user_turn`` returns a plain dict (JSON-serialisable) carrying the
cleaned utterance plus the sampling metadata that produced it, so downstream
stages can log/reproduce exactly how each user turn was generated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


# System prompt that pins the model into the "user" role. The assistant under
# test is a separate system; here Qwen must only ever speak *as the user*.
SYSTEM_PROMPT = """You are a user in a spoken multi-turn conversation with a voice assistant.
You are not the assistant.

Rules:
- Produce only the next user utterance.
- Keep the utterance suitable for speech synthesis.
- Do not mention that you are an AI or simulator.
- Do not answer your own question.
- Maintain the topic unless the dialogue naturally drifts.
- Use short or medium-length spoken sentences.
- If the assistant response was confusing, contradictory, hallucinated, or irrelevant, ask a clarification or correct it.
- Output plain text only, with no thinking text.
"""


@dataclass
class QwenUserSimulator:
    """Thin wrapper around the vLLM OpenAI-compatible endpoint.

    Attributes:
        base_url: vLLM OpenAI-compatible base URL, e.g. ``http://127.0.0.1:8001/v1``.
        model: Served model name (the ``--served-model-name`` passed to vLLM).
        temperature: Sampling temperature; >0 keeps user turns diverse across runs.
        top_p: Nucleus-sampling cutoff.
        max_tokens: Hard cap on the generated utterance length.
    """

    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "qwen3-32b-user-sim"
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 128

    def __post_init__(self) -> None:
        # vLLM ignores the API key but the SDK still requires a truthy value.
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")

    def generate_user_turn(
        self,
        topic: str,
        history: list[dict[str, str]],
        turn_id: int,
        user_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate the next simulated user utterance.

        Args:
            topic: The conversation topic the user should stay anchored to.
            history: Prior turns as ``{"role": "user"|"assistant", "text": ...}``
                dicts, oldest first.
            turn_id: 1-based index of the user turn we are about to produce
                (used only as context in the prompt).
            user_profile: Optional persona key/values (e.g. ``{"mood": "curious"}``)
                rendered into the prompt to steer the simulated user.

        Returns:
            Dict with the cleaned ``utterance`` plus the ``turn_id``/``topic`` and
            the sampling params used, for reproducible logging downstream.
        """
        prompt = self._build_prompt(topic, history, turn_id, user_profile)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            # Qwen3 supports a "thinking" mode that emits <think>...</think>
            # scratchpad tokens. We disable it so the response is a clean,
            # speakable utterance with no reasoning leakage.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        utterance = clean_user_utterance(response.choices[0].message.content or "")
        return {
            "utterance": utterance,
            "turn_id": turn_id,
            "topic": topic,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    @staticmethod
    def _build_prompt(
        topic: str,
        history: list[dict[str, str]],
        turn_id: int,
        user_profile: dict[str, Any] | None,
    ) -> str:
        """Render the dialogue state into the single user-role message.

        The conversation history is flattened into a readable transcript so the
        model has full context, then capped with an explicit instruction to emit
        only the next user line.
        """
        lines = [f"Topic: {topic}", f"Turn: {turn_id}", ""]
        if user_profile:
            profile = ", ".join(f"{k}={v}" for k, v in sorted(user_profile.items()))
            lines.extend([f"User profile: {profile}", ""])
        lines.append("Conversation so far:")
        if not history:
            lines.append("(none yet)")
        else:
            # Only render well-formed user/assistant turns; skip anything else
            # (e.g. system notes) so the transcript stays clean.
            for item in history:
                role = item.get("role", "").strip().lower()
                text = item.get("text", "").strip()
                if role in {"user", "assistant"} and text:
                    lines.append(f"{role.title()}: {text}")
        lines.extend(["", "Generate the next user utterance only."])
        return "\n".join(lines)


def clean_user_utterance(text: str) -> str:
    """Strip model artifacts so only the spoken utterance remains.

    Removes any leaked ``<think>`` blocks, a leading role label the model may
    echo (``User:`` / ``Assistant:`` / ``System:``), and surrounding quotes.
    """
    text = text.strip()
    # Drop any thinking scratchpad that slipped through despite enable_thinking=False.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Drop a leading "Role:" prefix if the model echoed one.
    text = re.sub(r"^\s*(User|Assistant|System)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip()
    return text
