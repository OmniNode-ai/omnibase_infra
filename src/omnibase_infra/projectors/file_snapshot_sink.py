# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""File-backed snapshot sink.

One file per (topic, key) pair; writes compact by overwriting. Used by
the dashboard fixture generator and by local dev adapters that do not
have a Kafka broker available.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class FileSnapshotSink:
    """Adapter that satisfies the publish-side of a ProtocolEventBusLike
    by writing JSON files to a filesystem root. Compaction semantics:
    publishing with the same (topic, key) OVERWRITES the prior file."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    async def publish(self, topic: str, key: str, value: BaseModel) -> None:
        topic_dir = self._root / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        (topic_dir / f"{key}.json").write_text(value.model_dump_json(indent=2))
