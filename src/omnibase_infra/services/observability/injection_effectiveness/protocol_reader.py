# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocol for injection effectiveness read operations.

Defines the ProtocolInjectionEffectivenessReader interface for querying
injection_effectiveness, latency_breakdowns, and pattern_hit_rates tables.

Design Decisions:
    - runtime_checkable: Enables isinstance() checks for duck typing
    - Async methods: All operations are async for non-blocking I/O
    - Typed models: Uses Pydantic models for type safety
    - Deterministic ordering: All queries use ORDER BY for reproducible results
      (created_at DESC for effectiveness/hit-rates, created_at ASC for latency)

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.services.observability.injection_effectiveness.models import (
        ModelInjectionEffectivenessQuery,
        ModelInjectionEffectivenessQueryResult,
        ModelInjectionEffectivenessRow,
        ModelLatencyBreakdownRow,
        ModelPatternHitRateRow,
    )


@runtime_checkable
class ProtocolInjectionEffectivenessReader(Protocol):
    """Protocol for injection effectiveness read operations.

    Provides query methods for all three injection effectiveness tables.
    Implementations must support pagination and deterministic ordering.

    Implementations:
        - ReaderInjectionEffectivenessPostgres: Production asyncpg implementation

    Example:
        >>> async def get_session_data(
        ...     reader: ProtocolInjectionEffectivenessReader,
        ...     session_id: UUID,
        ... ) -> ModelInjectionEffectivenessRow | None:
        ...     return await reader.query_by_session_id(session_id)
    """

    async def query_by_session_id(
        self,
        session_id: UUID,
        correlation_id: UUID | None = None,
    ) -> ModelInjectionEffectivenessRow | None:
        """Query a single session's injection effectiveness data.

        Args:
            session_id: Session identifier (primary key).
            correlation_id: Correlation ID for tracing (auto-generated if None).

        Returns:
            ModelInjectionEffectivenessRow if found, None otherwise.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def query(
        self,
        query: ModelInjectionEffectivenessQuery,
        correlation_id: UUID | None = None,
    ) -> ModelInjectionEffectivenessQueryResult:
        """Query with flexible filters, returns paginated results.

        Builds a dynamic WHERE clause from non-None fields of the query
        model. Returns paginated results with total_count metadata.

        Args:
            query: Query parameters with optional filters.
            correlation_id: Correlation ID for tracing (auto-generated if None).

        Returns:
            ModelInjectionEffectivenessQueryResult with pagination metadata.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def query_latency_breakdowns(
        self,
        session_id: UUID,
        correlation_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelLatencyBreakdownRow]:
        """Query latency breakdowns for a specific session.

        Returns per-prompt latency data ordered by created_at ascending
        (chronological prompt order).

        Args:
            session_id: Session identifier.
            correlation_id: Correlation ID for tracing (auto-generated if None).
            limit: Maximum rows to return (1-10000, default: 100).
            offset: Pagination offset (>= 0, default: 0).

        Returns:
            List of ModelLatencyBreakdownRow for the session.

        Raises:
            ValueError: If limit or offset is out of bounds.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def query_pattern_hit_rates(
        self,
        pattern_id: UUID | None = None,
        confident_only: bool = False,
        correlation_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelPatternHitRateRow]:
        """Query pattern hit rates, optionally filtered.

        Args:
            pattern_id: Filter by pattern ID (None = all patterns).
            confident_only: If True, only return patterns with confidence != NULL
                (sample_count >= minimum support threshold).
            correlation_id: Correlation ID for tracing (auto-generated if None).
            limit: Maximum rows to return (1-10000, default: 100).
            offset: Pagination offset (>= 0, default: 0).

        Returns:
            List of ModelPatternHitRateRow ordered by updated_at DESC.

        Raises:
            ValueError: If limit or offset is out of bounds.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...


__all__ = ["ProtocolInjectionEffectivenessReader"]
