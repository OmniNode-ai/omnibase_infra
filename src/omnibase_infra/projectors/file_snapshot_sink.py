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

    @staticmethod
    def _validate_segment(name: str, value: str) -> None:
        """Reject path-traversal / separator inputs for a single filesystem segment."""
        if value in {"", ".", ".."} or any(sep in value for sep in ("/", "\\")):
            raise ValueError(f"Invalid {name}: {value!r}")

    async def publish(self, topic: str, key: str, value: BaseModel) -> None:
        self._validate_segment("topic", topic)
        self._validate_segment("key", key)
        root = self._root.resolve()
        topic_dir = (root / topic).resolve()
        if not topic_dir.is_relative_to(root):
            raise ValueError(
                f"Resolved topic directory escapes snapshot root: {topic_dir}"
            )
        topic_dir.mkdir(parents=True, exist_ok=True)
        target = (topic_dir / f"{key}.json").resolve()
        if not target.is_relative_to(topic_dir):
            raise ValueError(
                f"Resolved snapshot path escapes topic directory: {target}"
            )
        target.write_text(value.model_dump_json(indent=2), encoding="utf-8")
