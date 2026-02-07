# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocol for validation event ledger repository operations.

This module defines the ProtocolValidationLedgerRepository interface for
validation event ledger persistence and replay. Implementations provide
append, query, and retention operations for the validation_event_ledger
table used by cross-repo validation runs.

Design Decisions:
    - runtime_checkable: Enables isinstance() checks for duck typing
    - Async methods: All operations are async for non-blocking I/O
    - Typed models: Uses Pydantic models for type safety
    - Idempotent writes: (kafka_topic, kafka_partition, kafka_offset) unique constraint
    - Deterministic replay: Ordering by kafka_partition, kafka_offset
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.models.validation_ledger import (
        ModelValidationLedgerAppendResult,
        ModelValidationLedgerEntry,
        ModelValidationLedgerQuery,
        ModelValidationLedgerReplayBatch,
    )


@runtime_checkable
class ProtocolValidationLedgerRepository(Protocol):
    """Protocol for validation event ledger persistence and replay.

    Provides append, query, and retention operations for the
    validation_event_ledger table. Implementations must support
    idempotent writes via (kafka_topic, kafka_partition, kafka_offset)
    unique constraint.

    Implementations:
        - PostgresValidationLedgerRepository: Production asyncpg implementation
        - MockValidationLedgerRepository: Test double for unit testing

    Example:
        >>> async def persist_event(
        ...     repo: ProtocolValidationLedgerRepository,
        ...     run_id: UUID,
        ...     envelope_bytes: bytes,
        ... ) -> ModelValidationLedgerAppendResult:
        ...     return await repo.append(
        ...         run_id=run_id,
        ...         repo_id="omnibase_core",
        ...         event_type="onex.evt.validation.cross-repo-run-started.v1",
        ...         event_version="v1",
        ...         occurred_at=datetime.now(UTC),
        ...         kafka_topic="onex.evt.validation.cross-repo-run-started.v1",
        ...         kafka_partition=0,
        ...         kafka_offset=42,
        ...         envelope_bytes=envelope_bytes,
        ...         envelope_hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        ...     )
    """

    async def append(
        self,
        *,
        run_id: UUID,
        repo_id: str,
        event_type: str,
        event_version: str,
        occurred_at: datetime,
        kafka_topic: str,
        kafka_partition: int,
        kafka_offset: int,
        envelope_bytes: bytes,
        envelope_hash: str,
    ) -> ModelValidationLedgerAppendResult:
        """Append a validation event to the ledger with idempotent write support.

        Uses INSERT ... ON CONFLICT DO NOTHING with the
        (kafka_topic, kafka_partition, kafka_offset) unique constraint.
        Duplicate events are detected without raising errors.

        Args:
            run_id: UUID of the validation run this event belongs to.
            repo_id: Repository identifier (e.g., "omnibase_core").
            event_type: Fully qualified event type name.
            event_version: Semantic version of the event schema.
            occurred_at: Timestamp when the event originally occurred.
            kafka_topic: Kafka topic the event was consumed from.
            kafka_partition: Kafka partition number.
            kafka_offset: Kafka offset within the partition.
            envelope_bytes: Raw envelope bytes stored as BYTEA.
            envelope_hash: SHA-256 hash of the envelope for integrity verification.

        Returns:
            ModelValidationLedgerAppendResult with:
                - success: True if operation completed without error
                - ledger_entry_id: UUID of created entry, None if duplicate
                - duplicate: True if ON CONFLICT was triggered

        Raises:
            RepositoryExecutionError: If database operation fails.
            RepositoryTimeoutError: If operation times out.
        """
        ...

    async def query_by_run_id(
        self,
        run_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelValidationLedgerEntry]:
        """Query entries for a specific validation run.

        Returns entries ordered by kafka_partition and kafka_offset for
        deterministic replay ordering.

        Args:
            run_id: The validation run UUID to query for.
            limit: Maximum number of entries to return (default: 100).
            offset: Number of entries to skip for pagination (default: 0).

        Returns:
            List of ModelValidationLedgerEntry matching the run_id,
            ordered by kafka_partition, kafka_offset for deterministic replay.

        Raises:
            RepositoryExecutionError: If database query fails.
            RepositoryTimeoutError: If query times out.
        """
        ...

    async def query(
        self,
        query: ModelValidationLedgerQuery,
    ) -> ModelValidationLedgerReplayBatch:
        """Query with flexible filters, returns paginated results.

        Builds a dynamic WHERE clause from the non-None fields of the query
        model. Returns a ModelValidationLedgerReplayBatch with pagination
        metadata (total_count, has_more).

        Args:
            query: Query parameters including optional filters for run_id,
                repo_id, event_type, start_time, end_time, and pagination.

        Returns:
            ModelValidationLedgerReplayBatch with:
                - entries: Matching ledger entries
                - total_count: Total matching rows (before pagination)
                - has_more: Whether more results exist beyond this page
                - query: The original query for reference

        Raises:
            RepositoryExecutionError: If database query fails.
            RepositoryTimeoutError: If query times out.
        """
        ...

    async def cleanup_expired(
        self,
        retention_days: int = 30,
        min_runs_per_repo: int = 25,
        batch_size: int = 1000,
    ) -> int:
        """Delete old entries respecting time-based and min-run retention.

        Implements a two-phase cleanup strategy:
        1. Identify protected run_ids (most recent min_runs_per_repo per repo)
        2. Delete entries older than retention_days that are NOT in protected runs

        Uses batched deletion to avoid long-running locks on large tables.

        Args:
            retention_days: Number of days to retain entries (default: 30).
            min_runs_per_repo: Minimum number of recent runs to preserve
                per repo_id, regardless of age (default: 25).
            batch_size: Number of records to delete per batch iteration
                (default: 1000).

        Returns:
            Total number of entries deleted across all batches.

        Raises:
            RepositoryExecutionError: If database operation fails.
            RepositoryTimeoutError: If cleanup times out.
        """
        ...


__all__ = ["ProtocolValidationLedgerRepository"]
