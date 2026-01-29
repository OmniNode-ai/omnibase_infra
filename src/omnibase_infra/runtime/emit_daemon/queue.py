# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Bounded Event Queue with Disk Spool for Hook Event Daemon.

This module provides a bounded in-memory queue with disk spool overflow
for buffering events when Kafka is slow or unavailable.

Queue Behavior:
    1. Events are first added to the in-memory queue
    2. When memory queue is full, events overflow to disk spool
    3. When disk spool is full (by message count or bytes), oldest events are dropped
    4. Dequeue prioritizes memory queue, then disk spool (FIFO ordering)

Disk Spool Format:
    - Directory: configurable (default: ~/.omniclaude/emit-spool/)
    - Files: {timestamp}_{event_id}.json (one event per file)
    - Sorted by filename for FIFO ordering

Concurrency Safety:
    This implementation is coroutine-safe using asyncio.Lock.
    Note: This is coroutine-safe, not thread-safe.

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP

.. versionadded:: 0.2.6
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)


class ModelQueuedEvent(BaseModel):
    """An event waiting to be published.

    Represents a single event in the bounded queue, with metadata
    for deduplication, routing, and retry tracking.

    Attributes:
        event_id: Unique identifier for deduplication (UUID string).
        event_type: The type/name of the event.
        topic: Kafka topic to publish to.
        payload: Event payload data.
        partition_key: Optional partition key for Kafka ordering.
        queued_at: UTC timestamp when the event was queued.
        retry_count: Number of publish retry attempts.

    Example:
        >>> from datetime import UTC, datetime
        >>> event = ModelQueuedEvent(
        ...     event_id="550e8400-e29b-41d4-a716-446655440000",
        ...     event_type="hook.event",
        ...     topic="claude-code-hook-events",
        ...     payload={"action": "test"},
        ...     queued_at=datetime.now(UTC),
        ... )
        >>> event.retry_count
        0
    """

    model_config = ConfigDict(
        strict=False,  # Allow coercion for JSON deserialization
        frozen=False,  # Allow retry_count mutation
        extra="forbid",
        from_attributes=True,
    )

    # ONEX_EXCLUDE: string_id - event_id is string for JSON serialization compatibility
    event_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for deduplication (UUID string).",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="The type/name of the event.",
    )
    topic: str = Field(
        ...,
        min_length=1,
        description="Kafka topic to publish to.",
    )
    payload: JsonType = Field(
        ...,
        description="Event payload data (JSON-compatible value).",
    )
    partition_key: str | None = Field(
        default=None,
        description="Optional partition key for Kafka ordering.",
    )
    queued_at: datetime = Field(
        ...,
        description="UTC timestamp when the event was queued.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of publish retry attempts.",
    )

    @field_validator("queued_at", mode="before")
    @classmethod
    def ensure_utc_aware(cls, v: object) -> object:
        """Ensure queued_at is timezone-aware (UTC).

        Args:
            v: The input value.

        Returns:
            UTC-aware datetime if input is datetime, otherwise unchanged input.
        """
        if not isinstance(v, datetime):
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        if v.utcoffset() == timedelta(0):
            if v.tzinfo is not UTC:
                return v.replace(tzinfo=UTC)
            return v
        return v.astimezone(UTC)


class BoundedEventQueue:
    """Bounded in-memory queue with disk spool overflow.

    This queue manages event buffering with configurable limits for both
    in-memory storage and disk spool. When limits are exceeded, the oldest
    spooled events are dropped to make room for new events.

    Attributes:
        max_memory_queue: Maximum events in memory queue.
        max_spool_messages: Maximum events in disk spool.
        max_spool_bytes: Maximum total bytes for disk spool.
        spool_dir: Directory for disk spool files.

    Overflow Behavior:
        1. If memory queue full -> spool to disk
        2. If spool full (messages OR bytes) -> drop oldest, then spool new

    Example:
        >>> import asyncio
        >>> from pathlib import Path
        >>>
        >>> async def example():
        ...     queue = BoundedEventQueue(
        ...         max_memory_queue=10,
        ...         max_spool_messages=100,
        ...     )
        ...     # Load any previously spooled events
        ...     await queue.load_spool()
        ...
        ...     # Enqueue an event
        ...     event = ModelQueuedEvent(
        ...         event_id="abc-123",
        ...         event_type="test",
        ...         topic="test-topic",
        ...         payload={"key": "value"},
        ...         queued_at=datetime.now(UTC),
        ...     )
        ...     success = await queue.enqueue(event)
        ...
        ...     # Dequeue for processing
        ...     next_event = await queue.dequeue()
    """

    def __init__(
        self,
        max_memory_queue: int = 100,
        max_spool_messages: int = 1000,
        max_spool_bytes: int = 10_485_760,  # 10 MB
        spool_dir: Path | None = None,
    ) -> None:
        """Initialize queue with limits.

        Args:
            max_memory_queue: Maximum number of events in memory queue.
                Defaults to 100.
            max_spool_messages: Maximum number of events in disk spool.
                Defaults to 1000.
            max_spool_bytes: Maximum total bytes for disk spool files.
                Defaults to 10 MB (10,485,760 bytes).
            spool_dir: Directory for disk spool files.
                Defaults to ~/.omniclaude/emit-spool/
        """
        self._max_memory_queue = max_memory_queue
        self._max_spool_messages = max_spool_messages
        self._max_spool_bytes = max_spool_bytes
        self._spool_dir = spool_dir or (Path.home() / ".omniclaude" / "emit-spool")

        # In-memory queue (FIFO)
        self._memory_queue: deque[ModelQueuedEvent] = deque()

        # Spool tracking
        self._spool_files: list[Path] = []  # Sorted by filename (FIFO order)
        self._spool_bytes: int = 0

        # Concurrency lock
        self._lock = asyncio.Lock()

        # Ensure spool directory exists
        self._ensure_spool_dir()

    def _ensure_spool_dir(self) -> None:
        """Ensure the spool directory exists."""
        try:
            self._spool_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                f"Failed to create spool directory {self._spool_dir}: {e}. "
                "Disk spool will be unavailable."
            )

    async def enqueue(self, event: ModelQueuedEvent) -> bool:
        """Add event to queue.

        Events are first added to the in-memory queue. If the memory queue
        is full, the event is spooled to disk. If the disk spool is also
        full, the oldest spooled event is dropped before adding the new one.

        Args:
            event: The event to queue.

        Returns:
            True if the event was queued (in memory or spool).
            False if the event could not be queued due to errors.

        Note:
            This method never raises exceptions. File I/O errors are logged
            and result in False being returned.
        """
        async with self._lock:
            # Try memory queue first
            if len(self._memory_queue) < self._max_memory_queue:
                self._memory_queue.append(event)
                logger.debug(
                    f"Event {event.event_id} queued in memory "
                    f"(memory: {len(self._memory_queue)}/{self._max_memory_queue})"
                )
                return True

            # Memory full - check if spooling is disabled
            if self._max_spool_messages == 0 or self._max_spool_bytes == 0:
                logger.warning(
                    f"Dropping event {event.event_id}: memory queue full "
                    f"({len(self._memory_queue)}/{self._max_memory_queue}) "
                    "and spooling is disabled (max_spool_messages=0 or max_spool_bytes=0)"
                )
                return False

            # Memory full, spool to disk
            return await self._spool_event(event)

    async def _spool_event(self, event: ModelQueuedEvent) -> bool:
        """Spool an event to disk.

        If the spool is full (by messages or bytes), drops the oldest
        event before adding the new one.

        Args:
            event: The event to spool.

        Returns:
            True if successfully spooled, False on error.

        Note:
            Caller must hold self._lock.
        """
        # Defensive check: if spooling is disabled, don't attempt to spool
        # (This should be checked by caller, but verify here for safety)
        if self._max_spool_messages == 0 or self._max_spool_bytes == 0:
            logger.debug(f"Spooling disabled, cannot spool event {event.event_id}")
            return False

        # Serialize event
        try:
            event_json = event.model_dump_json()
            event_bytes = len(event_json.encode("utf-8"))
        except Exception:
            logger.exception("Failed to serialize event %s", event.event_id)
            return False

        # Check if we need to drop oldest to make room
        while (
            len(self._spool_files) >= self._max_spool_messages
            or self._spool_bytes + event_bytes > self._max_spool_bytes
        ) and self._spool_files:
            await self._drop_oldest_spool()

        # Write to spool
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}_{event.event_id}.json"
        filepath = self._spool_dir / filename

        try:
            filepath.write_text(event_json, encoding="utf-8")
            self._spool_files.append(filepath)
            self._spool_bytes += event_bytes
            logger.debug(
                f"Event {event.event_id} spooled to disk "
                f"(spool: {len(self._spool_files)}/{self._max_spool_messages}, "
                f"bytes: {self._spool_bytes}/{self._max_spool_bytes})"
            )
            return True
        except OSError:
            logger.exception("Failed to write spool file %s", filepath)
            return False

    async def _drop_oldest_spool(self) -> None:
        """Drop the oldest spooled event.

        Note:
            Caller must hold self._lock.
        """
        if not self._spool_files:
            return

        oldest = self._spool_files.pop(0)
        try:
            file_size = oldest.stat().st_size
            oldest.unlink()
            self._spool_bytes -= file_size
            # Extract event_id from filename (timestamp_eventid.json)
            event_id = (
                oldest.stem.split("_", 1)[1] if "_" in oldest.stem else oldest.stem
            )
            logger.warning(
                f"Dropping oldest spooled event {event_id} due to spool overflow"
            )
        except OSError:
            logger.exception("Failed to delete oldest spool file %s", oldest)
            # Still remove from tracking to avoid infinite loop
            self._spool_bytes = max(0, self._spool_bytes)

    async def dequeue(self) -> ModelQueuedEvent | None:
        """Get next event to publish.

        Prioritizes memory queue, then disk spool. Returns None if both
        are empty.

        Returns:
            The next event to publish, or None if queue is empty.
        """
        async with self._lock:
            # Try memory queue first
            if self._memory_queue:
                event = self._memory_queue.popleft()
                logger.debug(
                    f"Dequeued event {event.event_id} from memory "
                    f"(remaining: {len(self._memory_queue)})"
                )
                return event

            # Try disk spool
            if self._spool_files:
                return await self._dequeue_from_spool()

            return None

    async def _dequeue_from_spool(self) -> ModelQueuedEvent | None:
        """Dequeue the next event from disk spool.

        Note:
            Caller must hold self._lock.

        Returns:
            The dequeued event, or None on error.
        """
        if not self._spool_files:
            return None

        filepath = self._spool_files.pop(0)
        try:
            # Read and parse
            content = filepath.read_text(encoding="utf-8")
            event = ModelQueuedEvent.model_validate_json(content)

            # Update byte tracking
            file_size = len(content.encode("utf-8"))
            self._spool_bytes -= file_size
        except OSError:
            logger.exception("Failed to read spool file %s", filepath)
            return None
        except Exception:
            logger.exception("Failed to parse spool file %s", filepath)
            # Delete corrupted file
            try:
                filepath.unlink()
            except OSError:
                pass
            return None

        # Delete file separately - event is already successfully parsed
        # If unlink fails, the event is still returned (not lost)
        try:
            filepath.unlink()
        except OSError:
            logger.warning(
                "Failed to delete spool file %s after successful dequeue - "
                "orphan file remains on disk",
                filepath,
            )
            # Event is still returned - not lost

        logger.debug(
            f"Dequeued event {event.event_id} from spool "
            f"(remaining spool: {len(self._spool_files)})"
        )
        return event

    async def peek(self) -> ModelQueuedEvent | None:
        """Peek at next event without removing it.

        Returns:
            The next event that would be dequeued, or None if empty.
        """
        async with self._lock:
            # Check memory queue
            if self._memory_queue:
                return self._memory_queue[0]

            # Check disk spool
            if self._spool_files:
                filepath = self._spool_files[0]
                try:
                    content = filepath.read_text(encoding="utf-8")
                    return ModelQueuedEvent.model_validate_json(content)
                except Exception:
                    logger.exception("Failed to peek at spool file %s", filepath)
                    return None

            return None

    def memory_size(self) -> int:
        """Number of events in memory queue (approximate).

        Returns:
            Count of events currently in the in-memory queue.

        Warning:
            This method does NOT acquire the lock. The returned value may be
            inconsistent during concurrent enqueue/dequeue operations. Use
            :meth:`memory_size_locked` when an accurate count is required.

        Note:
            Suitable for monitoring, logging, and approximate status reporting
            where eventual consistency is acceptable.
        """
        return len(self._memory_queue)

    async def memory_size_locked(self) -> int:
        """Number of events in memory queue (thread-safe).

        Acquires the queue lock before reading the size, ensuring a consistent
        value even during concurrent operations.

        Returns:
            Accurate count of events currently in the in-memory queue.

        Note:
            Use this method when an accurate count is required (e.g., for
            capacity decisions or precise status reporting). For approximate
            monitoring where lock contention is undesirable, use :meth:`memory_size`.
        """
        async with self._lock:
            return len(self._memory_queue)

    def spool_size(self) -> int:
        """Number of events in disk spool (approximate).

        Returns:
            Count of events currently in the disk spool.

        Warning:
            This method does NOT acquire the lock. The returned value may be
            inconsistent during concurrent enqueue/dequeue operations. Use
            :meth:`spool_size_locked` when an accurate count is required.

        Note:
            Suitable for monitoring, logging, and approximate status reporting
            where eventual consistency is acceptable.
        """
        return len(self._spool_files)

    async def spool_size_locked(self) -> int:
        """Number of events in disk spool (thread-safe).

        Acquires the queue lock before reading the size, ensuring a consistent
        value even during concurrent operations.

        Returns:
            Accurate count of events currently in the disk spool.

        Note:
            Use this method when an accurate count is required (e.g., for
            capacity decisions or precise status reporting). For approximate
            monitoring where lock contention is undesirable, use :meth:`spool_size`.
        """
        async with self._lock:
            return len(self._spool_files)

    def total_size(self) -> int:
        """Total events in memory and spool (approximate).

        Returns:
            Total count of events across memory and disk spool.

        Warning:
            This method does NOT acquire the lock. The returned value may be
            inconsistent during concurrent operations since it reads memory
            and spool sizes separately. Use :meth:`total_size_locked` when
            an accurate count is required.
        """
        return self.memory_size() + self.spool_size()

    async def total_size_locked(self) -> int:
        """Total events in memory and spool (thread-safe).

        Acquires the queue lock before reading sizes, ensuring a consistent
        total even during concurrent operations.

        Returns:
            Accurate total count of events across memory and disk spool.

        Note:
            Use this method when an accurate count is required. For approximate
            monitoring where lock contention is undesirable, use :meth:`total_size`.
        """
        async with self._lock:
            return len(self._memory_queue) + len(self._spool_files)

    async def drain_to_spool(self) -> int:
        """Move all memory events to spool for graceful shutdown.

        This should be called during graceful shutdown to persist
        in-memory events before the process exits.

        Returns:
            Number of events successfully moved to spool.

        Note:
            If spooling is disabled (max_spool_messages=0 or max_spool_bytes=0),
            this method will log a warning and return 0 without draining any
            events. Events in memory will be lost.
        """
        async with self._lock:
            # Check if spooling is disabled
            if self._max_spool_messages == 0 or self._max_spool_bytes == 0:
                memory_count = len(self._memory_queue)
                if memory_count > 0:
                    logger.warning(
                        f"Spooling is disabled (max_spool_messages=0 or max_spool_bytes=0). "
                        f"{memory_count} events in memory will be lost during shutdown."
                    )
                return 0

            count = 0
            while self._memory_queue:
                event = self._memory_queue.popleft()
                if await self._spool_event(event):
                    count += 1
                else:
                    logger.error(f"Failed to spool event {event.event_id} during drain")
            logger.info(f"Drained {count} events from memory to spool")
            return count

    async def load_spool(self) -> int:
        """Load spooled events on startup.

        Scans the spool directory for existing event files and
        rebuilds the spool tracking state. Files are sorted by
        filename for FIFO ordering.

        Returns:
            Number of events loaded from spool.
        """
        async with self._lock:
            self._spool_files.clear()
            self._spool_bytes = 0

            if not self._spool_dir.exists():
                logger.debug(f"Spool directory {self._spool_dir} does not exist")
                return 0

            try:
                # Find all .json files and sort by name (FIFO order)
                files = sorted(self._spool_dir.glob("*.json"))
                for filepath in files:
                    try:
                        file_size = filepath.stat().st_size
                        self._spool_files.append(filepath)
                        self._spool_bytes += file_size
                    except OSError as e:
                        logger.warning(f"Failed to stat spool file {filepath}: {e}")

                count = len(self._spool_files)
                if count > 0:
                    logger.info(
                        f"Loaded {count} events from spool ({self._spool_bytes} bytes)"
                    )
                return count
            except OSError:
                logger.exception("Failed to scan spool directory")
                return 0


__all__: list[str] = ["BoundedEventQueue", "ModelQueuedEvent"]
