"""Import shim for SALMONN short_v2 evaluation with lhotse fbank.

``evaluate_hf.py`` imports ``kaldifeat`` at module load time even when
``speech_encoder_type=transformer_v2`` makes it use Lhotse features instead.
This stub lets that import succeed in environments that have ``kaldifst`` but
not ``kaldifeat``. It intentionally fails if code tries to instantiate the
unused kaldifeat fbank path.
"""


class FbankOptions:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("kaldifeat is not installed; use lhotse_fbank=True")


class Fbank:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("kaldifeat is not installed; use lhotse_fbank=True")

