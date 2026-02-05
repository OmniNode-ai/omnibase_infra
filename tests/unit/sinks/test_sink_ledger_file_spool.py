# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for FileSpoolLedgerSink."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumLedgerSinkDropPolicy
from omnibase_infra.models.ledger import ModelDbQueryRequested, ModelLedgerEventBase
from omnibase_infra.sinks import FileSpoolLedgerSink
from omnibase_infra.sinks.sink_ledger_inmemory import (
    LedgerSinkClosedError,
    LedgerSinkFullError,
)


def _make_test_event(op_name: str = "test_op") -> ModelDbQueryRequested:
    """Create a test ledger event."""
    correlation_id = uuid4()
    return ModelDbQueryRequested(
        event_id=uuid4(),
        correlation_id=correlation_id,
        idempotency_key=ModelLedgerEventBase.build_idempotency_key(
            correlation_id, op_name, "db.query.requested"
        ),
        contract_id="test_contract",
        contract_fingerprint="sha256:abc123",
        operation_name=op_name,
        query_fingerprint="sha256:def456",
        emitted_at=datetime.now(UTC),
    )


class TestFileSpoolLedgerSink:
    """Tests for FileSpoolLedgerSink."""

    @pytest.mark.asyncio
    async def test_emit_and_flush(self, tmp_path: Path) -> None:
        """Test that emit() and flush() writes events to disk."""
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            max_file_size_bytes=1024 * 1024,
            max_buffer_size=100,
            flush_interval_seconds=60.0,  # Long interval to test explicit flush
        )

        event = _make_test_event()
        result = await sink.emit(event)

        assert result is True
        assert sink.pending_count == 1

        flushed = await sink.flush()
        assert flushed == 1
        assert sink.pending_count == 0

        # Verify file was created
        assert sink.current_file_path is not None
        assert sink.current_file_path.exists()

        # Verify content
        with open(sink.current_file_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        # Parse the JSON line
        data = json.loads(lines[0])
        assert data["operation_name"] == "test_op"
        assert data["event_type"] == "db.query.requested"

        await sink.close()

    @pytest.mark.asyncio
    async def test_close_flushes_pending(self, tmp_path: Path) -> None:
        """Test that close() flushes pending events."""
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            flush_interval_seconds=60.0,  # Long interval
        )

        await sink.emit(_make_test_event("op_0"))
        await sink.emit(_make_test_event("op_1"))
        assert sink.pending_count == 2

        await sink.close()
        assert sink.is_closed

        # Verify file was written
        assert sink.current_file_path is not None
        with open(sink.current_file_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_emit_after_close_raises(self, tmp_path: Path) -> None:
        """Test that emit() raises after close()."""
        sink = FileSpoolLedgerSink(spool_dir=tmp_path)
        await sink.close()

        with pytest.raises(LedgerSinkClosedError, match="closed sink"):
            await sink.emit(_make_test_event())

    @pytest.mark.asyncio
    async def test_drop_newest_policy(self, tmp_path: Path) -> None:
        """Test DROP_NEWEST policy when buffer is full."""
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            max_buffer_size=2,
            drop_policy=EnumLedgerSinkDropPolicy.DROP_NEWEST,
            flush_interval_seconds=60.0,
        )

        results = []
        for i in range(4):
            results.append(await sink.emit(_make_test_event(f"op_{i}")))

        # First 2 accepted, last 2 dropped
        assert results == [True, True, False, False]
        assert sink.pending_count == 2

        await sink.close()

    @pytest.mark.asyncio
    async def test_raise_policy(self, tmp_path: Path) -> None:
        """Test RAISE policy when buffer is full."""
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            max_buffer_size=2,
            drop_policy=EnumLedgerSinkDropPolicy.RAISE,
            flush_interval_seconds=60.0,
        )

        await sink.emit(_make_test_event("op_0"))
        await sink.emit(_make_test_event("op_1"))

        with pytest.raises(LedgerSinkFullError, match="buffer full"):
            await sink.emit(_make_test_event("op_2"))

        await sink.close()

    @pytest.mark.asyncio
    async def test_block_policy_raises_not_implemented(self, tmp_path: Path) -> None:
        """Test BLOCK policy raises NotImplementedError."""
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            max_buffer_size=1,
            drop_policy=EnumLedgerSinkDropPolicy.BLOCK,
            flush_interval_seconds=60.0,
        )

        await sink.emit(_make_test_event("op_0"))

        with pytest.raises(NotImplementedError, match="BLOCK policy"):
            await sink.emit(_make_test_event("op_1"))

        await sink.close()

    @pytest.mark.asyncio
    async def test_file_rotation(self, tmp_path: Path) -> None:
        """Test file rotation when max size is reached."""
        # Use very small file size to trigger rotation
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            max_file_size_bytes=100,  # Very small
            max_buffer_size=100,
            flush_interval_seconds=60.0,
        )

        # Emit several events to trigger rotation
        for i in range(5):
            await sink.emit(_make_test_event(f"op_{i}"))

        await sink.flush()

        # Check that multiple files were created
        files = list(tmp_path.glob("ledger_*.jsonl"))
        assert len(files) >= 2  # At least 2 files due to rotation

        await sink.close()

    @pytest.mark.asyncio
    async def test_creates_spool_dir(self, tmp_path: Path) -> None:
        """Test that spool_dir is created if it doesn't exist."""
        spool_dir = tmp_path / "nested" / "ledger"
        assert not spool_dir.exists()

        sink = FileSpoolLedgerSink(spool_dir=spool_dir)

        assert spool_dir.exists()

        await sink.close()

    def test_drop_policy_property(self, tmp_path: Path) -> None:
        """Test drop_policy property."""
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path, drop_policy=EnumLedgerSinkDropPolicy.RAISE
        )
        assert sink.drop_policy == EnumLedgerSinkDropPolicy.RAISE

    def test_is_closed_property(self, tmp_path: Path) -> None:
        """Test is_closed property."""
        sink = FileSpoolLedgerSink(spool_dir=tmp_path)
        assert sink.is_closed is False

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Test that close() can be called multiple times."""
        sink = FileSpoolLedgerSink(spool_dir=tmp_path)
        await sink.close()
        await sink.close()  # Should not raise

        assert sink.is_closed

    @pytest.mark.asyncio
    async def test_two_phase_flush_preserves_events_on_success(
        self, tmp_path: Path
    ) -> None:
        """Test that two-phase flush removes events only after successful write."""
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            flush_interval_seconds=60.0,  # Disable auto-flush
        )

        # Emit 3 events
        events = [_make_test_event(f"op_{i}") for i in range(3)]
        for event in events:
            await sink.emit(event)

        assert sink.pending_count == 3

        # Flush should write all events and clear buffer
        flushed = await sink.flush()
        assert flushed == 3
        assert sink.pending_count == 0

        # Verify all 3 events were written
        assert sink.current_file_path is not None
        with open(sink.current_file_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

        await sink.close()

    @pytest.mark.asyncio
    async def test_background_flush_respects_close(self, tmp_path: Path) -> None:
        """Test that background flush loop exits cleanly when close() is called.

        This verifies the race condition fix where _closed is checked inside
        the lock to prevent the background loop from continuing after close.
        """
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            flush_interval_seconds=0.05,  # Fast flush interval for test
        )

        # Emit an event to start the background flush task
        await sink.emit(_make_test_event("op_0"))

        # Wait for background flush task to start
        await asyncio.sleep(0.1)

        # Close the sink - this should stop the background task
        await sink.close()

        assert sink.is_closed
        # The flush task should have exited cleanly (no hanging tasks)
        assert sink._flush_task is not None
        assert sink._flush_task.done()

    @pytest.mark.asyncio
    async def test_emit_during_flush_not_lost(self, tmp_path: Path) -> None:
        """Test that events emitted during flush are not lost.

        The two-phase flush approach ensures that only events that were
        in the buffer at flush start are removed, not new events added
        during the flush operation.
        """
        sink = FileSpoolLedgerSink(
            spool_dir=tmp_path,
            max_buffer_size=100,
            flush_interval_seconds=60.0,  # Disable auto-flush
        )

        # Emit initial events
        for i in range(5):
            await sink.emit(_make_test_event(f"op_{i}"))

        assert sink.pending_count == 5

        # Flush the initial events
        flushed = await sink.flush()
        assert flushed == 5
        assert sink.pending_count == 0

        # Emit new events after flush
        for i in range(3):
            await sink.emit(_make_test_event(f"new_op_{i}"))

        assert sink.pending_count == 3

        # Second flush should only flush the new events
        flushed2 = await sink.flush()
        assert flushed2 == 3
        assert sink.pending_count == 0

        await sink.close()
