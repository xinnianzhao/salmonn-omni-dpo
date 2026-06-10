"""DPO-facing TTS interface.

A stable HTTP boundary around CosyVoice2: the server (:mod:`tts_interface.server`)
loads the model once and exposes ``/synthesize``; pipeline code talks to it
through :class:`TTSClient`. Heavy CosyVoice deps and model files stay outside
this repo.
"""

from .client import TTSClient, TTSRequestError
from .schemas import TTSRequest, TTSResponse

__all__ = ["TTSClient", "TTSRequestError", "TTSRequest", "TTSResponse"]
