# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""File-backed snapshot source.

Reads the compacted snapshot file for every key under a topic directory
and yields them as typed Pydantic models. Symmetric to FileSnapshotSink.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel


class FileSnapshotSource[S: BaseModel]:
    def __init__(self, root: Path, model: type[S]) -> None:
        self._root = root
        self._model = model

    async def read_all(self, topic: str) -> AsyncIterator[S]:
        # Block traversal: reject separators/parent segments and verify the
        # resolved directory stays under the snapshot root.
        if topic in {"", ".", ".."} or any(sep in topic for sep in ("/", "\\")):
            raise ValueError(f"Invalid topic: {topic!r}")
        root = self._root.resolve()
        topic_dir = (root / topic).resolve()
        if not topic_dir.is_relative_to(root):
            raise ValueError(
                f"Resolved topic directory escapes snapshot root: {topic_dir}"
            )
        if not topic_dir.is_dir():
            return
        for snapshot_file in topic_dir.glob("*.json"):
            yield self._model.model_validate_json(
                snapshot_file.read_text(encoding="utf-8")
            )
