# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for BoundedEventQueue with disk spool (OMN-1610).

This test suite validates the bounded event queue implementation:
- ModelQueuedEvent model validation and serialization
- Memory queue operations (enqueue, dequeue, peek)
- Memory queue limits and overflow to disk spool
- Disk spool operations (spooling, loading, file naming)
- Spool limits (message count, bytes, drop-oldest policy)
- Size tracking (memory_size, spool_size, total_size)
- Graceful shutdown (drain_to_spool)
- Dequeue priority (memory before spool)
- Concurrent access safety (asyncio.Lock)

Test Organization:
    - TestModelQueuedEvent: Event model validation and serialization
    - TestMemoryQueueBasicOperations: Basic enqueue/dequeue/peek
    - TestMemoryQueueLimits: Memory queue limit enforcement
    - TestDiskSpool: Spool file operations and loading
    - TestSpoolLimits: Spool message/byte limits and drop-oldest
    - TestTotalSize: Combined size tracking
    - TestDrainToSpool: Graceful shutdown drain
    - TestDequeuePriority: Memory-first dequeue ordering
    - TestEmptyQueue: Empty state handling
    - TestConcurrentAccess: Asyncio lock behavior

Coverage Goals:
    - >90% code coverage for queue module
    - All acceptance criteria from OMN-1610
    - All error paths tested
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.runtime.emit_daemon.queue import (
    BoundedEventQueue,
    ModelQueuedEvent,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_event() -> ModelQueuedEvent:
    """Create a sample queued event for testing.

    Returns:
        A valid ModelQueuedEvent instance.
    """
    return ModelQueuedEvent(
        event_id=str(uuid4()),
        event_type="test.event",
        topic="test-topic",
        payload={"key": "value", "count": 42},
        partition_key="partition-1",
        queued_at=datetime.now(UTC),
        retry_count=0,
    )


@pytest.fixture
def sample_event_factory():
    """Factory for creating unique sample events.

    Returns:
        Callable that creates new ModelQueuedEvent instances.
    """

    def _create(
        event_type: str = "test.event",
        topic: str = "test-topic",
        payload: dict | None = None,
        partition_key: str | None = None,
        retry_count: int = 0,
    ) -> ModelQueuedEvent:
        return ModelQueuedEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            topic=topic,
            payload=payload or {"key": "value"},
            partition_key=partition_key,
            queued_at=datetime.now(UTC),
            retry_count=retry_count,
        )

    return _create


@pytest.fixture
def queue_with_spool(tmp_path: Path) -> BoundedEventQueue:
    """Create a queue with a temporary spool directory.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        BoundedEventQueue configured with temp spool directory.
    """
    return BoundedEventQueue(
        max_memory_queue=10,
        max_spool_messages=100,
        max_spool_bytes=10_485_760,  # 10 MB
        spool_dir=tmp_path / "spool",
    )


@pytest.fixture
def small_queue(tmp_path: Path) -> BoundedEventQueue:
    """Create a queue with small limits for testing overflow.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        BoundedEventQueue with small memory/spool limits.
    """
    return BoundedEventQueue(
        max_memory_queue=3,
        max_spool_messages=5,
        max_spool_bytes=5000,  # 5 KB
        spool_dir=tmp_path / "spool",
    )


# ============================================================================
# TestModelQueuedEvent
# ============================================================================


class TestModelQueuedEvent:
    """Tests for ModelQueuedEvent Pydantic model."""

    def test_creates_valid_event_with_all_fields(self) -> None:
        """Event creation with all fields succeeds."""
        event_id = str(uuid4())
        queued_at = datetime.now(UTC)

        event = ModelQueuedEvent(
            event_id=event_id,
            event_type="hook.event",
            topic="claude-code-hook-events",
            payload={"action": "test", "data": {"nested": True}},
            partition_key="session-123",
            queued_at=queued_at,
            retry_count=3,
        )

        assert event.event_id == event_id
        assert event.event_type == "hook.event"
        assert event.topic == "claude-code-hook-events"
        assert event.payload == {"action": "test", "data": {"nested": True}}
        assert event.partition_key == "session-123"
        assert event.queued_at == queued_at
        assert event.retry_count == 3

    def test_creates_event_with_required_fields_only(self) -> None:
        """Event creation with only required fields succeeds."""
        event = ModelQueuedEvent(
            event_id="test-id",
            event_type="test.event",
            topic="test-topic",
            payload={"key": "value"},
            queued_at=datetime.now(UTC),
        )

        assert event.event_id == "test-id"
        assert event.partition_key is None
        assert event.retry_count == 0

    def test_validates_required_event_id(self) -> None:
        """Event creation fails with empty event_id."""
        with pytest.raises(ValueError, match="String should have at least 1 character"):
            ModelQueuedEvent(
                event_id="",
                event_type="test.event",
                topic="test-topic",
                payload={},
                queued_at=datetime.now(UTC),
            )

    def test_validates_required_event_type(self) -> None:
        """Event creation fails with empty event_type."""
        with pytest.raises(ValueError, match="String should have at least 1 character"):
            ModelQueuedEvent(
                event_id="test-id",
                event_type="",
                topic="test-topic",
                payload={},
                queued_at=datetime.now(UTC),
            )

    def test_validates_required_topic(self) -> None:
        """Event creation fails with empty topic."""
        with pytest.raises(ValueError, match="String should have at least 1 character"):
            ModelQueuedEvent(
                event_id="test-id",
                event_type="test.event",
                topic="",
                payload={},
                queued_at=datetime.now(UTC),
            )

    def test_validates_retry_count_non_negative(self) -> None:
        """Event creation fails with negative retry_count."""
        with pytest.raises(
            ValueError, match="Input should be greater than or equal to 0"
        ):
            ModelQueuedEvent(
                event_id="test-id",
                event_type="test.event",
                topic="test-topic",
                payload={},
                queued_at=datetime.now(UTC),
                retry_count=-1,
            )

    def test_serializes_to_json(self, sample_event: ModelQueuedEvent) -> None:
        """Event serializes to valid JSON string."""
        json_str = sample_event.model_dump_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["event_id"] == sample_event.event_id
        assert parsed["event_type"] == sample_event.event_type
        assert parsed["topic"] == sample_event.topic
        assert parsed["payload"] == sample_event.payload

    def test_deserializes_from_json(self, sample_event: ModelQueuedEvent) -> None:
        """Event deserializes from JSON string."""
        json_str = sample_event.model_dump_json()
        restored = ModelQueuedEvent.model_validate_json(json_str)

        assert restored.event_id == sample_event.event_id
        assert restored.event_type == sample_event.event_type
        assert restored.topic == sample_event.topic
        assert restored.payload == sample_event.payload
        assert restored.partition_key == sample_event.partition_key
        assert restored.retry_count == sample_event.retry_count

    def test_ensures_utc_timezone_for_naive_datetime(self) -> None:
        """Naive datetime is converted to UTC-aware."""
        naive_dt = datetime(2025, 1, 15, 10, 30, 0)
        event = ModelQueuedEvent(
            event_id="test-id",
            event_type="test.event",
            topic="test-topic",
            payload={},
            queued_at=naive_dt,
        )

        assert event.queued_at.tzinfo is UTC
        assert event.queued_at.year == 2025
        assert event.queued_at.month == 1
        assert event.queued_at.day == 15

    def test_converts_non_utc_timezone_to_utc(self) -> None:
        """Non-UTC timezone is converted to UTC."""
        # Create datetime in UTC+5 timezone
        utc_plus_5 = timezone(timedelta(hours=5))
        dt_plus_5 = datetime(2025, 1, 15, 15, 30, 0, tzinfo=utc_plus_5)

        event = ModelQueuedEvent(
            event_id="test-id",
            event_type="test.event",
            topic="test-topic",
            payload={},
            queued_at=dt_plus_5,
        )

        # Should be converted to UTC (15:30 UTC+5 = 10:30 UTC)
        assert event.queued_at.tzinfo is UTC
        assert event.queued_at.hour == 10

    def test_preserves_utc_timezone(self) -> None:
        """UTC-aware datetime is preserved as-is."""
        utc_dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        event = ModelQueuedEvent(
            event_id="test-id",
            event_type="test.event",
            topic="test-topic",
            payload={},
            queued_at=utc_dt,
        )

        assert event.queued_at.tzinfo is UTC
        assert event.queued_at == utc_dt

    def test_retry_count_is_mutable(self) -> None:
        """retry_count can be modified after creation."""
        event = ModelQueuedEvent(
            event_id="test-id",
            event_type="test.event",
            topic="test-topic",
            payload={},
            queued_at=datetime.now(UTC),
            retry_count=0,
        )

        event.retry_count = 5
        assert event.retry_count == 5


# ============================================================================
# TestMemoryQueueBasicOperations
# ============================================================================


class TestMemoryQueueBasicOperations:
    """Tests for basic memory queue operations."""

    @pytest.mark.asyncio
    async def test_enqueue_adds_to_memory_queue(
        self, queue_with_spool: BoundedEventQueue, sample_event: ModelQueuedEvent
    ) -> None:
        """enqueue() adds event to memory queue."""
        success = await queue_with_spool.enqueue(sample_event)

        assert success is True
        assert queue_with_spool.memory_size() == 1
        assert queue_with_spool.spool_size() == 0

    @pytest.mark.asyncio
    async def test_dequeue_returns_events_in_fifo_order(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """dequeue() returns events in FIFO order."""
        events = [sample_event_factory() for _ in range(3)]
        for event in events:
            await queue_with_spool.enqueue(event)

        for expected in events:
            result = await queue_with_spool.dequeue()
            assert result is not None
            assert result.event_id == expected.event_id

    @pytest.mark.asyncio
    async def test_peek_returns_next_event_without_removing(
        self, queue_with_spool: BoundedEventQueue, sample_event: ModelQueuedEvent
    ) -> None:
        """peek() returns next event without removing it."""
        await queue_with_spool.enqueue(sample_event)

        peeked = await queue_with_spool.peek()
        assert peeked is not None
        assert peeked.event_id == sample_event.event_id
        assert queue_with_spool.memory_size() == 1  # Not removed

        # Dequeue should return same event
        dequeued = await queue_with_spool.dequeue()
        assert dequeued is not None
        assert dequeued.event_id == sample_event.event_id
        assert queue_with_spool.memory_size() == 0

    @pytest.mark.asyncio
    async def test_memory_size_returns_correct_count(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """memory_size() returns correct count of events."""
        assert queue_with_spool.memory_size() == 0

        for i in range(5):
            await queue_with_spool.enqueue(sample_event_factory())
            assert queue_with_spool.memory_size() == i + 1

    @pytest.mark.asyncio
    async def test_multiple_enqueue_dequeue_cycles(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """Multiple enqueue/dequeue cycles work correctly."""
        # First cycle
        event1 = sample_event_factory()
        await queue_with_spool.enqueue(event1)
        result1 = await queue_with_spool.dequeue()
        assert result1 is not None
        assert result1.event_id == event1.event_id

        # Second cycle
        event2 = sample_event_factory()
        event3 = sample_event_factory()
        await queue_with_spool.enqueue(event2)
        await queue_with_spool.enqueue(event3)
        assert queue_with_spool.memory_size() == 2

        result2 = await queue_with_spool.dequeue()
        assert result2 is not None
        assert result2.event_id == event2.event_id
        assert queue_with_spool.memory_size() == 1


# ============================================================================
# TestMemoryQueueLimits
# ============================================================================


class TestMemoryQueueLimits:
    """Tests for memory queue limit enforcement."""

    @pytest.mark.asyncio
    async def test_respects_max_memory_queue_limit(
        self, small_queue: BoundedEventQueue, sample_event_factory
    ) -> None:
        """Queue respects max_memory_queue limit."""
        # Fill memory queue to limit (3)
        for _ in range(3):
            await small_queue.enqueue(sample_event_factory())

        assert small_queue.memory_size() == 3

        # Fourth event should go to spool, not exceed memory limit
        await small_queue.enqueue(sample_event_factory())
        assert small_queue.memory_size() == 3
        assert small_queue.spool_size() == 1

    @pytest.mark.asyncio
    async def test_overflow_goes_to_disk_spool(
        self, small_queue: BoundedEventQueue, sample_event_factory
    ) -> None:
        """Events overflow to disk spool when memory full."""
        # Fill memory queue
        for _ in range(3):
            await small_queue.enqueue(sample_event_factory())

        # Add more events - should go to spool
        for i in range(3):
            event = sample_event_factory()
            success = await small_queue.enqueue(event)
            assert success is True
            assert small_queue.spool_size() == i + 1

        assert small_queue.memory_size() == 3
        assert small_queue.spool_size() == 3

    @pytest.mark.asyncio
    async def test_spool_files_created_on_overflow(
        self, small_queue: BoundedEventQueue, sample_event_factory, tmp_path: Path
    ) -> None:
        """Spool files are created when memory overflows."""
        # Fill memory and overflow
        for _ in range(5):
            await small_queue.enqueue(sample_event_factory())

        spool_dir = tmp_path / "spool"
        spool_files = list(spool_dir.glob("*.json"))
        assert len(spool_files) == 2  # 5 - 3 = 2 in spool


# ============================================================================
# TestDiskSpool
# ============================================================================


class TestDiskSpool:
    """Tests for disk spool operations."""

    @pytest.mark.asyncio
    async def test_spooled_events_use_correct_naming(
        self, small_queue: BoundedEventQueue, sample_event_factory, tmp_path: Path
    ) -> None:
        """Spooled files use {timestamp}_{event_id}.json naming."""
        # Fill memory and cause overflow
        for _ in range(4):
            await small_queue.enqueue(sample_event_factory())

        spool_dir = tmp_path / "spool"
        spool_files = list(spool_dir.glob("*.json"))
        assert len(spool_files) == 1

        # Check naming pattern: YYYYMMDDHHMMSS{microseconds}_{event_id}.json
        filename = spool_files[0].name
        parts = filename.replace(".json", "").split("_", 1)
        assert len(parts) == 2
        timestamp_part = parts[0]
        event_id_part = parts[1]

        # Timestamp should be numeric
        assert timestamp_part.isdigit()
        assert len(timestamp_part) == 20  # YYYYMMDDHHMMSS + 6 microseconds

        # Event ID should be UUID-like
        assert len(event_id_part) == 36  # UUID length

    @pytest.mark.asyncio
    async def test_spool_size_returns_correct_count(
        self, small_queue: BoundedEventQueue, sample_event_factory
    ) -> None:
        """spool_size() returns correct count of spooled events."""
        assert small_queue.spool_size() == 0

        # Fill memory queue
        for _ in range(3):
            await small_queue.enqueue(sample_event_factory())

        # Add to spool
        for i in range(3):
            await small_queue.enqueue(sample_event_factory())
            assert small_queue.spool_size() == i + 1

    @pytest.mark.asyncio
    async def test_load_spool_loads_existing_files(
        self, tmp_path: Path, sample_event: ModelQueuedEvent
    ) -> None:
        """load_spool() loads existing spool files on startup."""
        spool_dir = tmp_path / "spool"
        spool_dir.mkdir(parents=True, exist_ok=True)

        # Write some spool files manually
        for i in range(3):
            event = ModelQueuedEvent(
                event_id=f"existing-{i}",
                event_type="test.event",
                topic="test-topic",
                payload={"index": i},
                queued_at=datetime.now(UTC),
            )
            filename = f"20250115100000{i:06d}_{event.event_id}.json"
            (spool_dir / filename).write_text(event.model_dump_json())

        # Create new queue and load spool
        queue = BoundedEventQueue(spool_dir=spool_dir)
        count = await queue.load_spool()

        assert count == 3
        assert queue.spool_size() == 3
        assert queue.memory_size() == 0

    @pytest.mark.asyncio
    async def test_load_spool_handles_empty_directory(self, tmp_path: Path) -> None:
        """load_spool() handles empty spool directory."""
        spool_dir = tmp_path / "spool"
        spool_dir.mkdir(parents=True, exist_ok=True)

        queue = BoundedEventQueue(spool_dir=spool_dir)
        count = await queue.load_spool()

        assert count == 0
        assert queue.spool_size() == 0

    @pytest.mark.asyncio
    async def test_load_spool_handles_nonexistent_directory(
        self, tmp_path: Path
    ) -> None:
        """load_spool() handles non-existent spool directory."""
        spool_dir = tmp_path / "nonexistent"
        queue = BoundedEventQueue(spool_dir=spool_dir)

        # load_spool should handle gracefully (directory will be created)
        count = await queue.load_spool()
        assert count == 0

    @pytest.mark.asyncio
    async def test_load_spool_sorts_files_for_fifo_order(self, tmp_path: Path) -> None:
        """load_spool() sorts files by name for FIFO ordering."""
        spool_dir = tmp_path / "spool"
        spool_dir.mkdir(parents=True, exist_ok=True)

        # Write files in reverse order of timestamps
        timestamps = ["20250115100003", "20250115100001", "20250115100002"]
        for ts in timestamps:
            event = ModelQueuedEvent(
                event_id=f"event-{ts}",
                event_type="test.event",
                topic="test-topic",
                payload={"ts": ts},
                queued_at=datetime.now(UTC),
            )
            filename = f"{ts}000000_{event.event_id}.json"
            (spool_dir / filename).write_text(event.model_dump_json())

        queue = BoundedEventQueue(spool_dir=spool_dir)
        await queue.load_spool()

        # Dequeue should return in sorted order
        event1 = await queue.dequeue()
        assert event1 is not None
        assert event1.event_id == "event-20250115100001"

        event2 = await queue.dequeue()
        assert event2 is not None
        assert event2.event_id == "event-20250115100002"


# ============================================================================
# TestSpoolLimits
# ============================================================================


class TestSpoolLimits:
    """Tests for spool limit enforcement."""

    @pytest.mark.asyncio
    async def test_respects_max_spool_messages_limit(
        self, tmp_path: Path, sample_event_factory
    ) -> None:
        """Queue respects max_spool_messages limit."""
        queue = BoundedEventQueue(
            max_memory_queue=2,
            max_spool_messages=3,
            max_spool_bytes=10_000_000,
            spool_dir=tmp_path / "spool",
        )

        # Fill memory (2) + spool (3)
        for _ in range(5):
            await queue.enqueue(sample_event_factory())

        assert queue.memory_size() == 2
        assert queue.spool_size() == 3

        # Adding more should trigger drop-oldest
        for _ in range(2):
            await queue.enqueue(sample_event_factory())

        # Spool should stay at limit (oldest dropped)
        assert queue.spool_size() == 3

    @pytest.mark.asyncio
    async def test_respects_max_spool_bytes_limit(
        self, tmp_path: Path, sample_event_factory
    ) -> None:
        """Queue respects max_spool_bytes limit."""
        # Create queue with small byte limit
        queue = BoundedEventQueue(
            max_memory_queue=1,
            max_spool_messages=1000,
            max_spool_bytes=500,  # Very small
            spool_dir=tmp_path / "spool",
        )

        # Fill memory
        await queue.enqueue(sample_event_factory())

        # Add events to spool until byte limit triggers drop
        for _ in range(5):
            event = sample_event_factory(payload={"data": "x" * 100})
            await queue.enqueue(event)

        # Spool size should be limited by bytes, not message count
        assert queue._spool_bytes <= 500

    @pytest.mark.asyncio
    async def test_drop_oldest_policy_when_spool_full(
        self, tmp_path: Path, sample_event_factory
    ) -> None:
        """Oldest spooled events are dropped when spool full."""
        queue = BoundedEventQueue(
            max_memory_queue=1,
            max_spool_messages=3,
            max_spool_bytes=10_000_000,
            spool_dir=tmp_path / "spool",
        )

        # Fill memory
        memory_event = sample_event_factory(event_type="memory")
        await queue.enqueue(memory_event)

        # Fill spool
        spool_events = []
        for i in range(3):
            event = sample_event_factory(event_type=f"spool-{i}")
            await queue.enqueue(event)
            spool_events.append(event)

        # Add new event - should drop oldest spool event (spool-0)
        new_event = sample_event_factory(event_type="new")
        await queue.enqueue(new_event)

        assert queue.spool_size() == 3

        # Dequeue memory event first
        result = await queue.dequeue()
        assert result is not None
        assert result.event_type == "memory"

        # Then spool events (spool-0 was dropped)
        result = await queue.dequeue()
        assert result is not None
        assert result.event_type == "spool-1"


# ============================================================================
# TestTotalSize
# ============================================================================


class TestTotalSize:
    """Tests for total_size() method."""

    @pytest.mark.asyncio
    async def test_total_size_returns_memory_plus_spool(
        self, small_queue: BoundedEventQueue, sample_event_factory
    ) -> None:
        """total_size() returns memory_size + spool_size."""
        assert small_queue.total_size() == 0

        # Add to memory
        for _ in range(2):
            await small_queue.enqueue(sample_event_factory())
        assert small_queue.total_size() == 2
        assert small_queue.memory_size() == 2
        assert small_queue.spool_size() == 0

        # Overflow to spool
        for _ in range(3):
            await small_queue.enqueue(sample_event_factory())

        assert small_queue.memory_size() == 3
        assert small_queue.spool_size() == 2
        assert small_queue.total_size() == 5

    @pytest.mark.asyncio
    async def test_total_size_updates_after_dequeue(
        self, small_queue: BoundedEventQueue, sample_event_factory
    ) -> None:
        """total_size() updates correctly after dequeue."""
        for _ in range(5):
            await small_queue.enqueue(sample_event_factory())

        assert small_queue.total_size() == 5

        await small_queue.dequeue()
        assert small_queue.total_size() == 4

        await small_queue.dequeue()
        assert small_queue.total_size() == 3


# ============================================================================
# TestDrainToSpool
# ============================================================================


class TestDrainToSpool:
    """Tests for drain_to_spool() method."""

    @pytest.mark.asyncio
    async def test_drain_moves_all_memory_events_to_spool(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """drain_to_spool() moves all memory events to spool."""
        # Add events to memory
        for _ in range(5):
            await queue_with_spool.enqueue(sample_event_factory())

        assert queue_with_spool.memory_size() == 5
        assert queue_with_spool.spool_size() == 0

        # Drain
        count = await queue_with_spool.drain_to_spool()

        assert count == 5
        assert queue_with_spool.memory_size() == 0
        assert queue_with_spool.spool_size() == 5

    @pytest.mark.asyncio
    async def test_drain_returns_count_of_drained_events(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """drain_to_spool() returns count of successfully drained events."""
        for _ in range(7):
            await queue_with_spool.enqueue(sample_event_factory())

        count = await queue_with_spool.drain_to_spool()
        assert count == 7

    @pytest.mark.asyncio
    async def test_drain_empties_memory_queue(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """Memory queue is empty after drain."""
        for _ in range(3):
            await queue_with_spool.enqueue(sample_event_factory())

        await queue_with_spool.drain_to_spool()

        assert queue_with_spool.memory_size() == 0
        # Peek should return from spool now
        peeked = await queue_with_spool.peek()
        assert peeked is not None

    @pytest.mark.asyncio
    async def test_drain_handles_empty_memory_queue(
        self, queue_with_spool: BoundedEventQueue
    ) -> None:
        """drain_to_spool() handles empty memory queue."""
        count = await queue_with_spool.drain_to_spool()
        assert count == 0
        assert queue_with_spool.memory_size() == 0
        assert queue_with_spool.spool_size() == 0

    @pytest.mark.asyncio
    async def test_drained_events_preserve_order(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """Drained events maintain FIFO order when read back."""
        events = []
        for i in range(5):
            event = sample_event_factory(event_type=f"event-{i}")
            await queue_with_spool.enqueue(event)
            events.append(event)

        await queue_with_spool.drain_to_spool()

        # Dequeue should return in original order
        for expected in events:
            result = await queue_with_spool.dequeue()
            assert result is not None
            assert result.event_id == expected.event_id


# ============================================================================
# TestDequeuePriority
# ============================================================================


class TestDequeuePriority:
    """Tests for dequeue priority (memory before spool)."""

    @pytest.mark.asyncio
    async def test_memory_dequeued_before_spool(
        self, tmp_path: Path, sample_event_factory
    ) -> None:
        """Memory queue is dequeued before spool."""
        queue = BoundedEventQueue(
            max_memory_queue=2,
            spool_dir=tmp_path / "spool",
        )

        # Add events to memory
        mem_event1 = sample_event_factory(event_type="memory-1")
        mem_event2 = sample_event_factory(event_type="memory-2")
        await queue.enqueue(mem_event1)
        await queue.enqueue(mem_event2)

        # Overflow to spool
        spool_event = sample_event_factory(event_type="spool-1")
        await queue.enqueue(spool_event)

        assert queue.memory_size() == 2
        assert queue.spool_size() == 1

        # Dequeue should return memory events first
        result1 = await queue.dequeue()
        assert result1 is not None
        assert result1.event_type == "memory-1"

        result2 = await queue.dequeue()
        assert result2 is not None
        assert result2.event_type == "memory-2"

        # Then spool
        result3 = await queue.dequeue()
        assert result3 is not None
        assert result3.event_type == "spool-1"

    @pytest.mark.asyncio
    async def test_spool_accessed_only_when_memory_empty(
        self, tmp_path: Path, sample_event_factory
    ) -> None:
        """Spool is only accessed when memory queue is empty."""
        spool_dir = tmp_path / "spool"
        queue = BoundedEventQueue(
            max_memory_queue=1,
            spool_dir=spool_dir,
        )

        # Add to memory
        mem_event = sample_event_factory(event_type="memory")
        await queue.enqueue(mem_event)

        # Overflow to spool
        spool_event = sample_event_factory(event_type="spool")
        await queue.enqueue(spool_event)

        # Verify spool file exists
        assert len(list(spool_dir.glob("*.json"))) == 1

        # Dequeue memory event - spool file should still exist
        result = await queue.dequeue()
        assert result is not None
        assert result.event_type == "memory"
        assert len(list(spool_dir.glob("*.json"))) == 1

        # Dequeue spool event - file should be deleted
        result = await queue.dequeue()
        assert result is not None
        assert result.event_type == "spool"
        assert len(list(spool_dir.glob("*.json"))) == 0


# ============================================================================
# TestEmptyQueue
# ============================================================================


class TestEmptyQueue:
    """Tests for empty queue handling."""

    @pytest.mark.asyncio
    async def test_dequeue_returns_none_when_empty(
        self, queue_with_spool: BoundedEventQueue
    ) -> None:
        """dequeue() returns None when queue is empty."""
        result = await queue_with_spool.dequeue()
        assert result is None

    @pytest.mark.asyncio
    async def test_peek_returns_none_when_empty(
        self, queue_with_spool: BoundedEventQueue
    ) -> None:
        """peek() returns None when queue is empty."""
        result = await queue_with_spool.peek()
        assert result is None

    @pytest.mark.asyncio
    async def test_sizes_are_zero_when_empty(
        self, queue_with_spool: BoundedEventQueue
    ) -> None:
        """All size methods return 0 for empty queue."""
        assert queue_with_spool.memory_size() == 0
        assert queue_with_spool.spool_size() == 0
        assert queue_with_spool.total_size() == 0

    @pytest.mark.asyncio
    async def test_dequeue_after_emptying_queue(
        self, queue_with_spool: BoundedEventQueue, sample_event: ModelQueuedEvent
    ) -> None:
        """dequeue() returns None after queue is emptied."""
        await queue_with_spool.enqueue(sample_event)
        await queue_with_spool.dequeue()

        result = await queue_with_spool.dequeue()
        assert result is None


# ============================================================================
# TestConcurrentAccess
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_uses_asyncio_lock(self, queue_with_spool: BoundedEventQueue) -> None:
        """Queue uses asyncio.Lock for synchronization."""
        assert hasattr(queue_with_spool, "_lock")
        assert isinstance(queue_with_spool._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_operations(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """Multiple concurrent enqueue operations work correctly."""

        async def enqueue_events(count: int):
            for _ in range(count):
                await queue_with_spool.enqueue(sample_event_factory())

        # Run multiple concurrent enqueue tasks
        await asyncio.gather(
            enqueue_events(5),
            enqueue_events(5),
            enqueue_events(5),
        )

        # All 15 events should be queued
        assert queue_with_spool.total_size() == 15

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_dequeue_operations(
        self, queue_with_spool: BoundedEventQueue, sample_event_factory
    ) -> None:
        """Concurrent enqueue and dequeue operations work correctly."""
        # Pre-fill queue
        for _ in range(10):
            await queue_with_spool.enqueue(sample_event_factory())

        dequeued_count = 0

        async def enqueue_events():
            for _ in range(5):
                await queue_with_spool.enqueue(sample_event_factory())
                await asyncio.sleep(0.001)

        async def dequeue_events():
            nonlocal dequeued_count
            for _ in range(5):
                result = await queue_with_spool.dequeue()
                if result:
                    dequeued_count += 1
                await asyncio.sleep(0.001)

        await asyncio.gather(
            enqueue_events(),
            dequeue_events(),
            dequeue_events(),
        )

        # Should have dequeued events and added new ones
        assert dequeued_count <= 10
        # Total should reflect operations (10 + 5 enqueued - dequeued)
        expected_remaining = 10 + 5 - dequeued_count
        assert queue_with_spool.total_size() == expected_remaining

    @pytest.mark.asyncio
    async def test_concurrent_peek_does_not_interfere(
        self, queue_with_spool: BoundedEventQueue, sample_event: ModelQueuedEvent
    ) -> None:
        """Concurrent peek operations do not interfere with queue state."""
        await queue_with_spool.enqueue(sample_event)

        async def peek_many():
            results = []
            for _ in range(10):
                result = await queue_with_spool.peek()
                results.append(result)
            return results

        # Multiple concurrent peeks
        all_results = await asyncio.gather(
            peek_many(),
            peek_many(),
            peek_many(),
        )

        # All peeks should return the same event
        for results in all_results:
            for result in results:
                assert result is not None
                assert result.event_id == sample_event.event_id

        # Queue should still have the event
        assert queue_with_spool.memory_size() == 1


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_enqueue_handles_serialization_error(
        self, queue_with_spool: BoundedEventQueue, tmp_path: Path
    ) -> None:
        """enqueue() handles serialization errors gracefully."""
        # Fill memory queue to force spooling
        queue = BoundedEventQueue(
            max_memory_queue=1,
            spool_dir=tmp_path / "spool",
        )
        await queue.enqueue(
            ModelQueuedEvent(
                event_id="first",
                event_type="test",
                topic="test",
                payload={},
                queued_at=datetime.now(UTC),
            )
        )

        # Create event with object that will fail serialization
        with patch.object(
            ModelQueuedEvent,
            "model_dump_json",
            side_effect=TypeError("Cannot serialize"),
        ):
            event = ModelQueuedEvent(
                event_id="bad",
                event_type="test",
                topic="test",
                payload={},
                queued_at=datetime.now(UTC),
            )
            # Should return False, not raise
            result = await queue.enqueue(event)
            assert result is False

    @pytest.mark.asyncio
    async def test_dequeue_handles_corrupted_spool_file(self, tmp_path: Path) -> None:
        """dequeue() handles corrupted spool file gracefully."""
        spool_dir = tmp_path / "spool"
        spool_dir.mkdir(parents=True, exist_ok=True)

        # Write corrupted spool file
        (spool_dir / "20250115100000000000_bad-event.json").write_text("not valid json")

        queue = BoundedEventQueue(spool_dir=spool_dir)
        await queue.load_spool()

        assert queue.spool_size() == 1

        # Dequeue should handle error and return None
        result = await queue.dequeue()
        assert result is None

        # Corrupted file should be deleted
        assert len(list(spool_dir.glob("*.json"))) == 0

    @pytest.mark.asyncio
    async def test_spool_handles_write_error(
        self, tmp_path: Path, sample_event_factory
    ) -> None:
        """Spooling handles write errors gracefully."""
        # Create a directory where we cannot write
        spool_dir = tmp_path / "readonly"
        spool_dir.mkdir(parents=True, exist_ok=True)

        queue = BoundedEventQueue(
            max_memory_queue=1,
            spool_dir=spool_dir,
        )

        # Fill memory
        await queue.enqueue(sample_event_factory())

        # Make spool dir read-only
        original_mode = spool_dir.stat().st_mode
        spool_dir.chmod(0o444)

        try:
            # Try to spool - should fail gracefully
            event = sample_event_factory()
            result = await queue.enqueue(event)
            assert result is False
        finally:
            # Restore permissions for cleanup
            spool_dir.chmod(original_mode)

    @pytest.mark.asyncio
    async def test_peek_handles_spool_read_error(self, tmp_path: Path) -> None:
        """peek() handles spool read errors gracefully."""
        spool_dir = tmp_path / "spool"
        spool_dir.mkdir(parents=True, exist_ok=True)

        # Write valid spool file
        event = ModelQueuedEvent(
            event_id="test-event",
            event_type="test",
            topic="test",
            payload={},
            queued_at=datetime.now(UTC),
        )
        filepath = spool_dir / "20250115100000000000_test-event.json"
        filepath.write_text(event.model_dump_json())

        queue = BoundedEventQueue(spool_dir=spool_dir)
        await queue.load_spool()

        # Make file unreadable
        original_mode = filepath.stat().st_mode
        filepath.chmod(0o000)

        try:
            # Peek should handle error
            result = await queue.peek()
            assert result is None
        finally:
            filepath.chmod(original_mode)


# ============================================================================
# TestSpoolDirectory
# ============================================================================


class TestSpoolDirectory:
    """Tests for spool directory handling."""

    def test_creates_spool_directory_on_init(self, tmp_path: Path) -> None:
        """Queue creates spool directory on initialization."""
        spool_dir = tmp_path / "new" / "nested" / "spool"
        assert not spool_dir.exists()

        BoundedEventQueue(spool_dir=spool_dir)

        assert spool_dir.exists()
        assert spool_dir.is_dir()

    def test_uses_default_spool_directory(self) -> None:
        """Queue uses default spool directory if not specified."""
        queue = BoundedEventQueue()
        expected_default = Path.home() / ".omniclaude" / "emit-spool"
        assert queue._spool_dir == expected_default

    @pytest.mark.asyncio
    async def test_handles_spool_dir_creation_failure(
        self, tmp_path: Path, caplog
    ) -> None:
        """Queue handles spool directory creation failure gracefully."""
        # Create a file where directory should be
        blocker = tmp_path / "spool"
        blocker.write_text("I am a file, not a directory")

        # Queue creation should log warning but not fail
        import logging

        with caplog.at_level(logging.WARNING):
            queue = BoundedEventQueue(spool_dir=blocker)

        # Spool operations will fail, but queue can still use memory
        assert queue._spool_dir == blocker


__all__ = [
    "TestModelQueuedEvent",
    "TestMemoryQueueBasicOperations",
    "TestMemoryQueueLimits",
    "TestDiskSpool",
    "TestSpoolLimits",
    "TestTotalSize",
    "TestDrainToSpool",
    "TestDequeuePriority",
    "TestEmptyQueue",
    "TestConcurrentAccess",
    "TestErrorHandling",
    "TestSpoolDirectory",
]
