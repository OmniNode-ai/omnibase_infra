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
        topic_dir = self._root / topic
        if not topic_dir.is_dir():
            return
        for snapshot_file in topic_dir.glob("*.json"):
            yield self._model.model_validate_json(snapshot_file.read_text())
