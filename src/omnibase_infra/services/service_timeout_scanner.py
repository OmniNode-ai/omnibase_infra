# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Timeout Scanner for querying overdue registration entities.

This scanner queries the registration projection for nodes that have:
- Passed their ack_deadline (for ack timeout detection)
- Passed their liveness_deadline (for liveness expiry detection)
- Not yet had a timeout event emitted (emission marker is NULL)

The scanner is used by the orchestrator during RuntimeTick processing
to identify nodes requiring timeout decision events.

Coroutine Safety:
    This scanner is stateless and delegates all database operations to the
    ProjectionReaderRegistration, which handles coroutine safety and circuit
    breaker protection.

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)

logger = logging.getLogger(__name__)


class ModelTimeoutQueryResult(BaseModel):
    """Result of a timeout query - nodes requiring timeout events.

    This model captures the results of both ack and liveness timeout
    queries, along with metadata about the query execution.

    The orchestrator uses this to emit timeout decision events for each
    entity in the result lists.

    Attributes:
        ack_timeouts: Nodes with overdue ack_deadline requiring timeout events.
            These are nodes in ACCEPTED or AWAITING_ACK state that have not
            acknowledged within the deadline.
        liveness_expirations: Active nodes with overdue liveness_deadline
            requiring expiry events. These are nodes in ACTIVE state that
            have missed their liveness deadline.
        query_time: The 'now' used for the query (from RuntimeTick).
            This is the injected time, not the system clock.
        query_duration_ms: Query execution time in milliseconds.
            Useful for performance monitoring and alerting.

    Example:
        >>> result = await scanner.find_overdue_entities(now=tick.now)
        >>> for proj in result.ack_timeouts:
        ...     emit_ack_timeout_event(proj.entity_id)
        >>> for proj in result.liveness_expirations:
        ...     emit_liveness_expired_event(proj.entity_id)
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    ack_timeouts: list[ModelRegistrationProjection] = Field(
        default_factory=list,
        description="Nodes with overdue ack_deadline requiring timeout events",
    )
    liveness_expirations: list[ModelRegistrationProjection] = Field(
        default_factory=list,
        description="Active nodes with overdue liveness_deadline requiring expiry events",
    )
    query_time: datetime = Field(
        ...,
        description="The 'now' used for query (from RuntimeTick)",
    )
    query_duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Query execution time in milliseconds",
    )

    @property
    def total_overdue_count(self) -> int:
        """Return total count of overdue entities across both categories."""
        return len(self.ack_timeouts) + len(self.liveness_expirations)

    @property
    def has_overdue_entities(self) -> bool:
        """Check if any overdue entities were found."""
        return self.total_overdue_count > 0


class ServiceTimeoutScanner:
    """Scanner for querying registration projections for timeout candidates.

    This scanner provides a high-level interface for the orchestrator to
    query for nodes requiring timeout events. It delegates to the
    ProjectionReaderRegistration for actual database queries.

    The scanner inherits circuit breaker protection from the projection
    reader - when the reader's circuit breaker opens, queries will raise
    InfraUnavailableError.

    Design Notes:
        - All queries use injected 'now' (never system clock)
        - Batch size is configurable to limit memory usage
        - Query timing is tracked for observability
        - Circuit breaker protection is inherited from reader

    SQL Index Usage:
        The underlying queries leverage optimized partial indexes:
        - idx_registration_ack_timeout_scan: For ack deadline queries
        - idx_registration_liveness_timeout_scan: For liveness deadline queries

    Usage:
        >>> reader = ProjectionReaderRegistration(pool)
        >>> scanner = ServiceTimeoutScanner(container, reader)
        >>> result = await scanner.find_overdue_entities(now=tick.now)
        >>>
        >>> for projection in result.ack_timeouts:
        ...     # Emit NodeRegistrationAckTimedOut event
        ...     emit_ack_timeout_event(projection.entity_id)
        >>>
        >>> for projection in result.liveness_expirations:
        ...     # Emit NodeLivenessExpired event
        ...     emit_liveness_expired_event(projection.entity_id)

    Attributes:
        DEFAULT_BATCH_SIZE: Default limit for query results (100).
            Prevents memory exhaustion when many nodes are overdue.

    Raises:
        InfraConnectionError: If database connection fails
        InfraTimeoutError: If query times out
        InfraUnavailableError: If circuit breaker is open
        RuntimeHostError: For other database errors
    """

    DEFAULT_BATCH_SIZE: int = 100

    def __init__(
        self,
        container: ModelONEXContainer,
        projection_reader: ProjectionReaderRegistration,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the timeout scanner service.

        Args:
            container: ONEX container for dependency injection.
            projection_reader: The projection reader for database queries.
                Must be initialized with an asyncpg connection pool.
            batch_size: Maximum entities to return per query type.
                Defaults to DEFAULT_BATCH_SIZE (100).

        Example:
            >>> pool = await asyncpg.create_pool(dsn)
            >>> reader = ProjectionReaderRegistration(pool)
            >>> scanner = ServiceTimeoutScanner(container, reader)
        """
        self._container = container
        self._reader = projection_reader
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE

    @property
    def batch_size(self) -> int:
        """Return configured batch size for queries."""
        return self._batch_size

    async def find_overdue_entities(
        self,
        now: datetime,
        domain: str = "registration",
        correlation_id: UUID | None = None,
    ) -> ModelTimeoutQueryResult:
        """Find all entities requiring timeout events.

        Queries both ack and liveness timeout candidates in parallel using
        asyncio.gather for optimal performance.

        The returned result contains both categories of overdue entities
        along with query metadata for observability.

        Args:
            now: The current time from RuntimeTick (injected, not system clock).
                This ensures deterministic behavior during replay.
            domain: Domain namespace (default: "registration").
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            ModelTimeoutQueryResult with lists of nodes needing timeout/expiry
            events and query metadata.

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors

        Example:
            >>> now = datetime.now(UTC)  # In production, use tick.now
            >>> result = await scanner.find_overdue_entities(now=now)
            >>> print(f"Found {result.total_overdue_count} overdue entities")
            >>> print(f"Query took {result.query_duration_ms:.2f}ms")
        """
        corr_id = correlation_id or uuid4()
        start_time = time.perf_counter()

        logger.debug(
            "Querying overdue entities",
            extra={
                "query_time": now.isoformat(),
                "domain": domain,
                "correlation_id": str(corr_id),
                "batch_size": self._batch_size,
            },
        )

        # Query both timeout types in parallel for optimal performance
        ack_timeouts, liveness_expirations = await asyncio.gather(
            self._reader.get_overdue_ack_registrations(
                now=now,
                domain=domain,
                limit=self._batch_size,
                correlation_id=corr_id,
            ),
            self._reader.get_overdue_liveness_registrations(
                now=now,
                domain=domain,
                limit=self._batch_size,
                correlation_id=corr_id,
            ),
        )

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000.0

        logger.debug(
            "Overdue entity query completed",
            extra={
                "ack_timeout_count": len(ack_timeouts),
                "liveness_expiration_count": len(liveness_expirations),
                "query_duration_ms": duration_ms,
                "correlation_id": str(corr_id),
            },
        )

        # Warn operators when batch limit is reached, indicating more entities may be pending.
        # This helps operators understand when the system is under high load and may need
        # multiple ticks to process all overdue entities.
        ack_at_limit = len(ack_timeouts) >= self._batch_size
        liveness_at_limit = len(liveness_expirations) >= self._batch_size

        if ack_at_limit or liveness_at_limit:
            logger.warning(
                "Batch size limit reached - additional overdue entities may be pending",
                extra={
                    "batch_size": self._batch_size,
                    "ack_timeout_count": len(ack_timeouts),
                    "ack_at_limit": ack_at_limit,
                    "liveness_expiration_count": len(liveness_expirations),
                    "liveness_at_limit": liveness_at_limit,
                    "correlation_id": str(corr_id),
                },
            )

        return ModelTimeoutQueryResult(
            ack_timeouts=ack_timeouts,
            liveness_expirations=liveness_expirations,
            query_time=now,
            query_duration_ms=duration_ms,
        )

    async def find_ack_timeouts(
        self,
        now: datetime,
        domain: str = "registration",
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find nodes with overdue ack deadlines.

        Queries for nodes where:
        - ack_deadline < now
        - ack_timeout_emitted_at IS NULL
        - current_state IN (ACCEPTED, AWAITING_ACK)

        This query leverages the idx_registration_ack_timeout_scan partial
        index for efficient scanning.

        Args:
            now: The current time from RuntimeTick (injected, not system clock).
            domain: Domain namespace (default: "registration").
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            List of projections for nodes needing ack timeout events.

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors

        Example:
            >>> overdue = await scanner.find_ack_timeouts(now=tick.now)
            >>> for proj in overdue:
            ...     print(f"Node {proj.entity_id} missed ack deadline")
        """
        corr_id = correlation_id or uuid4()

        logger.debug(
            "Querying ack timeouts",
            extra={
                "query_time": now.isoformat(),
                "domain": domain,
                "correlation_id": str(corr_id),
            },
        )

        return await self._reader.get_overdue_ack_registrations(
            now=now,
            domain=domain,
            limit=self._batch_size,
            correlation_id=corr_id,
        )

    async def find_liveness_expirations(
        self,
        now: datetime,
        domain: str = "registration",
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find active nodes with overdue liveness deadlines.

        Queries for nodes where:
        - liveness_deadline < now
        - liveness_timeout_emitted_at IS NULL
        - current_state = ACTIVE

        This query leverages the idx_registration_liveness_timeout_scan
        partial index for efficient scanning.

        Args:
            now: The current time from RuntimeTick (injected, not system clock).
            domain: Domain namespace (default: "registration").
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            List of projections for nodes needing liveness expiry events.

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors

        Example:
            >>> expired = await scanner.find_liveness_expirations(now=tick.now)
            >>> for proj in expired:
            ...     print(f"Node {proj.entity_id} missed liveness deadline")
        """
        corr_id = correlation_id or uuid4()

        logger.debug(
            "Querying liveness expirations",
            extra={
                "query_time": now.isoformat(),
                "domain": domain,
                "correlation_id": str(corr_id),
            },
        )

        return await self._reader.get_overdue_liveness_registrations(
            now=now,
            domain=domain,
            limit=self._batch_size,
            correlation_id=corr_id,
        )


__all__: list[str] = ["ModelTimeoutQueryResult", "ServiceTimeoutScanner"]
