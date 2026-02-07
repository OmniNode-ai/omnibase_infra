# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocol definition for validation event ledger repository operations.

This module defines the ProtocolValidationLedgerRepository interface for
validation event storage, retrieval, and retention management.

Design Decisions:
    - runtime_checkable: Enables isinstance() checks for duck typing
    - Async methods: All operations are async for non-blocking I/O
    - Typed models: Uses Pydantic models for type safety
    - Replay ordering: By (kafka_partition, kafka_offset) within topic

Ticket: OMN-1908
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

    from omnibase_infra.models.validation_ledger import (
        ModelValidationLedgerAppendResult,
        ModelValidationLedgerEntry,
        ModelValidationLedgerQuery,
        ModelValidationLedgerReplayBatch,
    )


@runtime_checkable
class ProtocolValidationLedgerRepository(Protocol):
    """Protocol for validation event ledger persistence operations.

    This protocol defines the interface for appending validation events
    to the domain-specific ledger, querying by run_id or flexible filters,
    and managing retention.

    Implementations must provide idempotent append operations via the
    (kafka_topic, kafka_partition, kafka_offset) unique constraint.

    Methods:
        append: Idempotent write of a validation event to the ledger.
        query_by_run_id: Retrieve all events for a specific validation run.
        query: Flexible query with optional filters and pagination.
        cleanup_expired: Remove old entries respecting retention policy.
    """

    async def append(
        self,
        entry: ModelValidationLedgerEntry,
    ) -> ModelValidationLedgerAppendResult:
        """Append a validation event to the ledger with idempotent write support.

        Uses INSERT ... ON CONFLICT DO NOTHING with the
        (kafka_topic, kafka_partition, kafka_offset) unique constraint.

        Args:
            entry: Validation ledger entry to persist.

        Returns:
            ModelValidationLedgerAppendResult with success, entry_id,
            and duplicate flag.

        Raises:
            RepositoryExecutionError: If database operation fails.
        """
        ...

    async def query_by_run_id(
        self,
        run_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelValidationLedgerEntry]:
        """Query validation ledger entries by run ID.

        Returns entries ordered by (kafka_partition, kafka_offset) for
        correct replay ordering within the run.

        Args:
            run_id: The validation run ID to search for.
            limit: Maximum entries to return (default: 100).
            offset: Number of entries to skip for pagination.

        Returns:
            List of entries ordered by Kafka offset for replay.
        """
        ...

    async def query(
        self,
        query: ModelValidationLedgerQuery,
    ) -> ModelValidationLedgerReplayBatch:
        """Execute a flexible query with optional filters and pagination.

        Builds WHERE clause from non-None query fields combined with AND.

        Args:
            query: Query parameters with optional filters.

        Returns:
            ModelValidationLedgerReplayBatch with entries, total_count,
            has_more, and the original query.
        """
        ...

    async def cleanup_expired(
        self,
        retention_days: int = 30,
        min_runs_per_repo: int = 25,
        max_deletions: int = 1000,
    ) -> int:
        """Remove expired validation ledger entries.

        Implements combined retention policy:
        1. Delete entries older than retention_days
        2. BUT preserve at least min_runs_per_repo distinct runs per repo

        Args:
            retention_days: Delete entries older than this many days.
            min_runs_per_repo: Minimum distinct runs to keep per repo.
            max_deletions: Maximum rows to delete per call (prevents lock contention).

        Returns:
            Number of entries deleted.
        """
        ...


__all__ = ["ProtocolValidationLedgerRepository"]
