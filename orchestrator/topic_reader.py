"""Topic readers for large JSON/JSONL source files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_topics(path: str | Path, *, limit: int | None = None, offset: int = 0) -> Iterator[dict[str, str]]:
    """Yield ``{"topic": ..., "source_id": ...}`` records from JSONL or JSON arrays.

    The current trivia file is a large JSON array. This reader streams it with
    ``json.JSONDecoder.raw_decode`` to avoid loading the full 1.7 GB file.
    """
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        yield from _iter_jsonl_topics(path, limit=limit, offset=offset)
        return
    yield from _iter_json_array_topics(path, limit=limit, offset=offset)


def _iter_jsonl_topics(path: Path, *, limit: int | None, offset: int) -> Iterator[dict[str, str]]:
    emitted = 0
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            topic = str(item.get("topic", "")).strip()
            if not topic:
                continue
            if seen < offset:
                seen += 1
                continue
            yield {"topic": topic, "source_id": item.get("id", f"{path.name}:{line_no}")}
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _iter_json_array_topics(path: Path, *, limit: int | None, offset: int) -> Iterator[dict[str, str]]:
    decoder = json.JSONDecoder()
    emitted = 0
    seen = 0
    buffer = ""
    pos = 0
    started = False

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk and pos >= len(buffer):
                return
            buffer += chunk
            while True:
                pos = _skip_ws(buffer, pos)
                if not started:
                    if pos >= len(buffer):
                        break
                    if buffer[pos] != "[":
                        raise ValueError(f"Expected JSON array in {path}")
                    started = True
                    pos += 1
                    continue
                pos = _skip_ws(buffer, pos)
                if pos >= len(buffer):
                    break
                if buffer[pos] == "]":
                    return
                if buffer[pos] == ",":
                    pos += 1
                    continue
                try:
                    item, next_pos = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break
                pos = next_pos
                topic = str(item.get("topic", "")).strip() if isinstance(item, dict) else ""
                if topic:
                    if seen < offset:
                        seen += 1
                    else:
                        yield {"topic": topic, "source_id": item.get("id", f"{path.name}:{seen}")}
                        emitted += 1
                        if limit is not None and emitted >= limit:
                            return
                if pos > 4 * 1024 * 1024:
                    buffer = buffer[pos:]
                    pos = 0
                    break


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos

