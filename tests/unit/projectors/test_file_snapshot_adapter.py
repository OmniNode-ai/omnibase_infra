# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for FileSnapshotSink + FileSnapshotSource.

Verifies the file-backed snapshot adapter pair:
- Sink writes compacted JSON per (topic, key) pair.
- Source reads back all snapshots under a topic and yields typed models.
- Sink OVERWRITES on republish with the same key (compaction semantics).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.projectors import FileSnapshotSink, FileSnapshotSource


class _Snap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str
    current_state: str


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_sink_writes_compacted_json(tmp_path: Path) -> None:
    sink = FileSnapshotSink(tmp_path)
    snap = _Snap(entity_id="alpha", current_state="active")

    await sink.publish("onex.evt.snap.v1", "alpha", snap)

    written = tmp_path / "onex.evt.snap.v1" / "alpha.json"
    assert written.exists()
    payload = json.loads(written.read_text())
    assert payload == {"entity_id": "alpha", "current_state": "active"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_source_reads_compacted_snapshots(tmp_path: Path) -> None:
    sink = FileSnapshotSink(tmp_path)
    await sink.publish(
        "onex.evt.snap.v1", "alpha", _Snap(entity_id="alpha", current_state="active")
    )
    await sink.publish(
        "onex.evt.snap.v1", "beta", _Snap(entity_id="beta", current_state="idle")
    )

    source = FileSnapshotSource[_Snap](tmp_path, _Snap)
    results = [snap async for snap in source.read_all("onex.evt.snap.v1")]

    assert len(results) == 2
    by_id = {snap.entity_id: snap.current_state for snap in results}
    assert by_id == {"alpha": "active", "beta": "idle"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_sink_compaction_overwrites_by_key(tmp_path: Path) -> None:
    sink = FileSnapshotSink(tmp_path)

    await sink.publish(
        "onex.evt.snap.v1", "alpha", _Snap(entity_id="alpha", current_state="pending")
    )
    await sink.publish(
        "onex.evt.snap.v1", "alpha", _Snap(entity_id="alpha", current_state="active")
    )

    source = FileSnapshotSource[_Snap](tmp_path, _Snap)
    results = [snap async for snap in source.read_all("onex.evt.snap.v1")]

    assert len(results) == 1
    assert results[0].entity_id == "alpha"
    assert results[0].current_state == "active"
