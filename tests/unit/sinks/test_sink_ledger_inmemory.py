# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for InMemoryLedgerSink."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

from omnibase_infra.models.ledger import ModelDbQueryRequested, ModelLedgerEventBase
from omnibase_infra.protocols import EnumLedgerSinkDropPolicy
from omnibase_infra.sinks import InMemoryLedgerSink
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


class TestInMemoryLedgerSink:
    """Tests for InMemoryLedgerSink."""

    @pytest.mark.asyncio
    async def test_emit_stores_event(self) -> None:
        """Test that emit() stores events in buffer."""
        sink = InMemoryLedgerSink()
        event = _make_test_event()

        result = await sink.emit(event)

        assert result is True
        assert sink.pending_count == 1
        assert len(sink.events) == 1
        assert sink.events[0] == event

    @pytest.mark.asyncio
    async def test_emit_multiple_events(self) -> None:
        """Test emitting multiple events."""
        sink = InMemoryLedgerSink()
        events = [_make_test_event(f"op_{i}") for i in range(5)]

        for event in events:
            await sink.emit(event)

        assert sink.pending_count == 5
        assert sink.events == events

    @pytest.mark.asyncio
    async def test_drop_oldest_policy(self) -> None:
        """Test DROP_OLDEST policy when buffer is full."""
        sink = InMemoryLedgerSink(
            max_size=3, drop_policy=EnumLedgerSinkDropPolicy.DROP_OLDEST
        )
        events = [_make_test_event(f"op_{i}") for i in range(5)]

        for event in events:
            result = await sink.emit(event)
            assert result is True  # Always accepted with DROP_OLDEST

        # Only last 3 events should remain
        assert sink.pending_count == 3
        assert sink.events[0].operation_name == "op_2"
        assert sink.events[2].operation_name == "op_4"

    @pytest.mark.asyncio
    async def test_drop_newest_policy(self) -> None:
        """Test DROP_NEWEST policy when buffer is full."""
        sink = InMemoryLedgerSink(
            max_size=3, drop_policy=EnumLedgerSinkDropPolicy.DROP_NEWEST
        )
        events = [_make_test_event(f"op_{i}") for i in range(5)]

        results = []
        for event in events:
            results.append(await sink.emit(event))

        # First 3 accepted, last 2 dropped
        assert results == [True, True, True, False, False]
        assert sink.pending_count == 3
        assert sink.events[0].operation_name == "op_0"
        assert sink.events[2].operation_name == "op_2"

    @pytest.mark.asyncio
    async def test_raise_policy(self) -> None:
        """Test RAISE policy when buffer is full."""
        sink = InMemoryLedgerSink(
            max_size=2, drop_policy=EnumLedgerSinkDropPolicy.RAISE
        )

        await sink.emit(_make_test_event("op_0"))
        await sink.emit(_make_test_event("op_1"))

        with pytest.raises(LedgerSinkFullError, match="buffer full"):
            await sink.emit(_make_test_event("op_2"))

    @pytest.mark.asyncio
    async def test_emit_after_close_raises(self) -> None:
        """Test that emit() raises after close()."""
        sink = InMemoryLedgerSink()
        await sink.close()

        with pytest.raises(LedgerSinkClosedError, match="closed sink"):
            await sink.emit(_make_test_event())

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """Test that close() can be called multiple times."""
        sink = InMemoryLedgerSink()
        await sink.close()
        await sink.close()  # Should not raise

        assert sink.is_closed

    @pytest.mark.asyncio
    async def test_flush_returns_count(self) -> None:
        """Test that flush() returns pending count."""
        sink = InMemoryLedgerSink()
        await sink.emit(_make_test_event("op_0"))
        await sink.emit(_make_test_event("op_1"))

        count = await sink.flush()

        assert count == 2

    def test_clear_removes_all_events(self) -> None:
        """Test that clear() removes all events."""
        sink = InMemoryLedgerSink()
        # Bypass async for this sync test
        sink._events.append(_make_test_event())
        sink._events.append(_make_test_event())

        sink.clear()

        assert sink.pending_count == 0
        assert len(sink.events) == 0

    def test_drop_policy_property(self) -> None:
        """Test drop_policy property."""
        sink = InMemoryLedgerSink(drop_policy=EnumLedgerSinkDropPolicy.RAISE)
        assert sink.drop_policy == EnumLedgerSinkDropPolicy.RAISE


class TestIdempotencyKey:
    """Tests for idempotency key generation."""

    def test_build_idempotency_key_format(self) -> None:
        """Test idempotency key has correct format."""
        correlation_id = uuid4()
        key = ModelLedgerEventBase.build_idempotency_key(
            correlation_id, "find_by_id", "db.query.requested"
        )

        assert f"{correlation_id}" in key
        assert "find_by_id" in key
        assert "db.query.requested" in key

    def test_idempotency_key_deterministic(self) -> None:
        """Test that same inputs produce same key."""
        correlation_id = uuid4()
        key1 = ModelLedgerEventBase.build_idempotency_key(
            correlation_id, "op", "event_type"
        )
        key2 = ModelLedgerEventBase.build_idempotency_key(
            correlation_id, "op", "event_type"
        )

        assert key1 == key2
