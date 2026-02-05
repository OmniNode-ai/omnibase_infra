# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""In-memory ledger sink for unit testing.

This sink stores events in memory for testing purposes only. Events are NOT
persisted and will be lost on process restart.

WARNING: This sink is NOT suitable for production use. For durable ledger
storage, use FileSpoolLedgerSink or a database-backed implementation.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

from omnibase_infra.protocols.protocol_ledger_sink import EnumLedgerSinkDropPolicy

if TYPE_CHECKING:
    from omnibase_infra.models.ledger import ModelLedgerEventBase


class LedgerSinkError(Exception):
    """Base exception for ledger sink errors."""


class LedgerSinkFullError(LedgerSinkError):
    """Raised when sink queue is full and policy is RAISE."""


class LedgerSinkClosedError(LedgerSinkError):
    """Raised when attempting to emit to a closed sink."""


class InMemoryLedgerSink:
    """In-memory ledger sink for unit testing.

    This sink stores events in a bounded deque for testing purposes.
    Events can be inspected via the `events` property.

    WARNING:
        NOT DURABLE. Events are lost on process restart.
        Use FileSpoolLedgerSink for production.

    Attributes:
        max_size: Maximum number of events to buffer.
        drop_policy: Policy when buffer is full.

    Example:
        >>> sink = InMemoryLedgerSink(max_size=1000)
        >>> await sink.emit(event)
        >>> assert len(sink.events) == 1
        >>> await sink.close()
    """

    __slots__ = (
        "_closed",
        "_drop_policy",
        "_events",
        "_lock",
        "_max_size",
    )

    def __init__(
        self,
        max_size: int = 10000,
        drop_policy: EnumLedgerSinkDropPolicy = EnumLedgerSinkDropPolicy.DROP_OLDEST,
    ) -> None:
        """Initialize the in-memory sink.

        Args:
            max_size: Maximum number of events to buffer (default: 10000).
            drop_policy: Policy when buffer is full (default: DROP_OLDEST).
        """
        self._max_size = max_size
        self._drop_policy = drop_policy
        self._events: deque[ModelLedgerEventBase] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._closed = False

    async def emit(self, event: ModelLedgerEventBase) -> bool:
        """Emit a ledger event to the in-memory buffer.

        Args:
            event: Ledger event to emit.

        Returns:
            True if event was accepted.
            False if event was dropped due to DROP_NEWEST policy.

        Raises:
            LedgerSinkClosedError: If sink is closed.
            LedgerSinkFullError: If buffer is full and policy is RAISE.
            NotImplementedError: If drop_policy is BLOCK (not supported in test sink).

        Note:
            With DROP_OLDEST policy (default), this method always returns True
            when the sink is open, but if the buffer is full, the oldest event
            is silently evicted to make room for the new event. Callers cannot
            detect when old events are dropped; use DROP_NEWEST or RAISE policies
            if drop notification is required.
        """
        async with self._lock:
            # Check closed INSIDE lock to prevent race with close()
            if self._closed:
                raise LedgerSinkClosedError("Cannot emit to closed sink")
            if len(self._events) >= self._max_size:
                if self._drop_policy == EnumLedgerSinkDropPolicy.DROP_NEWEST:
                    return False
                elif self._drop_policy == EnumLedgerSinkDropPolicy.RAISE:
                    raise LedgerSinkFullError(
                        f"Sink buffer full ({self._max_size} events)"
                    )
                elif self._drop_policy == EnumLedgerSinkDropPolicy.BLOCK:
                    # BLOCK policy requires proper condition variable implementation
                    # which is not provided in this test-only sink
                    raise NotImplementedError(
                        "BLOCK policy is not supported in InMemoryLedgerSink. "
                        "Use DROP_OLDEST, DROP_NEWEST, or RAISE for testing."
                    )
                # DROP_OLDEST: deque with maxlen handles this automatically

            self._events.append(event)
            return True

    async def flush(self) -> int:
        """Flush is a no-op for in-memory sink.

        Returns:
            Number of events currently in buffer.
        """
        return len(self._events)

    async def close(self) -> None:
        """Close the sink.

        After close(), emit() will raise LedgerSinkClosedError.
        """
        self._closed = True

    @property
    def drop_policy(self) -> EnumLedgerSinkDropPolicy:
        """Get the configured drop policy."""
        return self._drop_policy

    @property
    def is_closed(self) -> bool:
        """Check if the sink is closed."""
        return self._closed

    @property
    def pending_count(self) -> int:
        """Get the number of events in the buffer."""
        return len(self._events)

    @property
    def events(self) -> list[ModelLedgerEventBase]:
        """Get all events in the buffer (for testing).

        Returns:
            List of events in emission order (oldest first).
        """
        return list(self._events)

    def clear(self) -> None:
        """Clear all events from the buffer (for testing)."""
        self._events.clear()


__all__ = [
    "InMemoryLedgerSink",
    "LedgerSinkClosedError",
    "LedgerSinkError",
    "LedgerSinkFullError",
]
