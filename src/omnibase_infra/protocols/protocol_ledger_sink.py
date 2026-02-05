# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocol definition for ledger sink operations.

This module defines the ProtocolLedgerSink interface for emitting ledger events
from runtime components. Sinks accept events asynchronously and are responsible
for durability guarantees.

Design Decisions:
    - runtime_checkable: Enables isinstance() checks for duck typing
    - Async methods: All operations are async to match runtime call() signatures
    - Non-blocking: Implementations must not block the caller; use bounded queues
    - Drop policy: Implementations must define explicit behavior when queue full
    - Cross-runtime: Protocol lives in top-level protocols/ for reuse by
      DB, HTTP, tool, and orchestrator runtimes

Latency Budget:
    Target p95 < 2ms for local sink operations. Implementations exceeding this
    budget must degrade gracefully (drop with metric, not block).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from omnibase_infra.enums import EnumLedgerSinkDropPolicy

if TYPE_CHECKING:
    from omnibase_infra.models.ledger import ModelLedgerEventBase


@runtime_checkable
class ProtocolLedgerSink(Protocol):
    """Protocol for ledger event sinks.

    This protocol defines the interface for emitting ledger events from runtime
    components. Implementations handle durability, batching, and delivery.

    Implementations:
        - FileSpoolLedgerSink: Durable append-only JSONL files with rotation
        - InMemoryLedgerSink: Test-only in-memory buffer (not durable)
        - PostgresLedgerSink: Direct database writes (future)
        - KafkaLedgerSink: Kafka producer integration (future)

    Concurrency Safety:
        All implementations MUST be safe for concurrent coroutine access.
        Use asyncio.Lock or asyncio.Queue for coordination.

    Example:
        >>> async def execute_query(
        ...     sink: ProtocolLedgerSink,
        ...     event: ModelDbQueryRequested,
        ... ) -> None:
        ...     await sink.emit(event)
    """

    async def emit(self, event: ModelLedgerEventBase) -> bool:
        """Emit a ledger event to the sink.

        This method must be non-blocking in the common case (queue not full).
        When the queue is full, behavior depends on the configured drop policy.

        Args:
            event: Ledger event to emit. Must be a subclass of ModelLedgerEventBase
                with all required envelope fields populated.

        Returns:
            True if event was accepted (queued or written).
            False if event was dropped due to policy (DROP_OLDEST/DROP_NEWEST).

        Raises:
            LedgerSinkFullError: If drop_policy is RAISE and queue is full.
            LedgerSinkError: If sink is closed or in error state.

        Performance:
            Target p95 < 2ms. Implementations must not perform synchronous I/O
            in the hot path; use buffering and background flush.
        """
        ...

    async def flush(self) -> int:
        """Flush pending events to durable storage.

        Forces immediate write of buffered events. Use sparingly as this
        may block until I/O completes.

        Returns:
            Number of events flushed.

        Raises:
            LedgerSinkError: If flush fails (I/O error, sink closed).
        """
        ...

    async def close(self) -> None:
        """Close the sink, flushing any pending events.

        After close(), emit() will raise LedgerSinkError.
        This method is idempotent (safe to call multiple times).

        Raises:
            LedgerSinkError: If final flush fails (events may be lost).
        """
        ...

    @property
    def drop_policy(self) -> EnumLedgerSinkDropPolicy:
        """Get the configured drop policy for this sink."""
        ...

    @property
    def is_closed(self) -> bool:
        """Check if the sink is closed."""
        ...

    @property
    def pending_count(self) -> int:
        """Get the number of events pending in the buffer/queue."""
        ...


__all__ = [
    "ProtocolLedgerSink",
]
