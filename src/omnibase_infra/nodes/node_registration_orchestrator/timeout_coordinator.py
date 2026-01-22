# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Timeout Coordinator for coordinating RuntimeTick events.

This coordinator is invoked when the orchestrator receives a RuntimeTick event.
It coordinates timeout detection and emission using the injected 'now' time.

Pattern:
    1. Receive RuntimeTick with injected 'now'
    2. Query for overdue entities using ServiceTimeoutScanner
    3. Emit timeout events using ServiceTimeoutEmitter
    4. Return result for observability

Design Decisions:
    - Uses tick.now for all time-based decisions (never system clock)
    - Propagates correlation_id from RuntimeTick for distributed tracing
    - Uses tick_id as causation_id for emitted events
    - Delegates to ServiceTimeoutEmitter for actual emission logic

Coroutine Safety:
    This coordinator is stateless and coroutine-safe for concurrent calls.
    Each call coordinates independently, delegating coroutine safety to
    the underlying services (ServiceTimeoutScanner, ServiceTimeoutEmitter).

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-888 (C1): Registration Orchestrator
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.services import (
    ServiceTimeoutEmitter,
    ServiceTimeoutScanner,
)

logger = logging.getLogger(__name__)


class ModelTimeoutCoordinationResult(BaseModel):
    """Result from coordinating a RuntimeTick for timeouts.

    Captures the complete result of timeout coordination for a single
    RuntimeTick event. Used for observability, metrics, and error tracking.

    Attributes:
        tick_id: ID of the coordinated RuntimeTick.
        tick_now: Injected 'now' from the tick (used for all time decisions).
        ack_timeouts_found: Number of ack timeout candidates found.
        liveness_expirations_found: Number of liveness expiration candidates found.
        ack_timeouts_emitted: Number of ack timeout events actually emitted.
        liveness_expirations_emitted: Number of liveness expiry events emitted.
        markers_updated: Number of projection markers updated.
        coordination_time_ms: Total coordinator coordination time in milliseconds.
        query_time_ms: Time spent querying for overdue entities.
        emission_time_ms: Time spent emitting events and updating markers.
        success: Whether coordination completed without errors.
        error: Error message if coordination failed.
        errors: Tuple of non-fatal errors encountered during coordination (immutable).

    Example:
        >>> result = await coordinator.coordinate(tick)
        >>> print(f"Coordinated tick {result.tick_id}")
        >>> print(f"Found {result.ack_timeouts_found} ack timeouts")
        >>> print(f"Emitted {result.ack_timeouts_emitted} events")
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    tick_id: UUID = Field(
        ...,
        description="ID of the coordinated RuntimeTick",
    )
    tick_now: datetime = Field(
        ...,
        description="Injected 'now' from the tick",
    )
    ack_timeouts_found: int = Field(
        default=0,
        ge=0,
        description="Number of ack timeout candidates found",
    )
    liveness_expirations_found: int = Field(
        default=0,
        ge=0,
        description="Number of liveness expiration candidates found",
    )
    ack_timeouts_emitted: int = Field(
        default=0,
        ge=0,
        description="Number of ack timeout events emitted",
    )
    liveness_expirations_emitted: int = Field(
        default=0,
        ge=0,
        description="Number of liveness expiry events emitted",
    )
    markers_updated: int = Field(
        default=0,
        ge=0,
        description="Number of projection markers updated",
    )
    coordination_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total coordinator coordination time in milliseconds",
    )
    query_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent querying for overdue entities",
    )
    emission_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent emitting events and updating markers",
    )
    success: bool = Field(
        default=True,
        description="Whether coordination completed without errors",
    )
    error: str | None = Field(
        default=None,
        description="Error message if coordination failed",
    )
    errors: tuple[str, ...] = Field(
        default=(),
        description="Tuple of non-fatal errors encountered (immutable for thread safety)",
    )

    @field_validator("errors", mode="before")
    @classmethod
    def _coerce_errors_to_tuple(cls, v: object) -> tuple[str, ...]:
        """Convert list/sequence to tuple for immutability.

        Args:
            v: The input value to coerce.

        Returns:
            A tuple of error strings.

        Raises:
            ValueError: If input is not a valid sequence type.
        """
        # NOTE: isinstance checks validate runtime type, but mypy cannot narrow
        # the generic Sequence type to tuple[str, ...] in this validator context.
        if isinstance(v, tuple):
            return v  # type: ignore[return-value]  # NOTE: runtime type validated above
        if isinstance(v, Sequence) and not isinstance(v, str | bytes):
            return tuple(v)  # type: ignore[return-value]  # NOTE: runtime type validated above
        raise ValueError(
            f"errors must be a tuple or Sequence (excluding str/bytes), "
            f"got {type(v).__name__}"
        )

    @property
    def total_found(self) -> int:
        """Return total count of timeout candidates found."""
        return self.ack_timeouts_found + self.liveness_expirations_found

    @property
    def total_emitted(self) -> int:
        """Return total count of events emitted."""
        return self.ack_timeouts_emitted + self.liveness_expirations_emitted

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred (fatal or non-fatal)."""
        return self.error is not None or len(self.errors) > 0


class TimeoutCoordinator:
    """Coordinator for RuntimeTick-triggered timeout coordination.

    This coordinator:
    1. Uses injected 'now' from RuntimeTick (never system clock)
    2. Queries for overdue entities via ServiceTimeoutScanner
    3. Emits timeout events via ServiceTimeoutEmitter
    4. Is restart-safe: only coordinates entities without emission markers

    The coordinator is designed to be invoked by the orchestrator when it
    receives a RuntimeTick event from the consumed events queue.

    Design Note:
        The coordinator delegates all business logic to the underlying services:
        - ServiceTimeoutScanner: Finds overdue entities
        - ServiceTimeoutEmitter: Emits events and updates markers

        This separation ensures testability and allows the services to be
        reused independently.

    Usage in orchestrator:
        >>> # Wire dependencies
        >>> timeout_query = ServiceTimeoutScanner(container, projection_reader)
        >>> timeout_emission = ServiceTimeoutEmitter(
        ...     container=container,
        ...     timeout_query=timeout_query,
        ...     event_bus=event_bus,
        ...     projector=projector,
        ... )
        >>> coordinator = TimeoutCoordinator(timeout_query, timeout_emission)
        >>>
        >>> # Coordinate RuntimeTick
        >>> result = await coordinator.coordinate(runtime_tick)
        >>> if not result.success:
        ...     log.error(f"Timeout coordination failed: {result.error}")

    Raises:
        InfraConnectionError: If database/Kafka connection fails
        InfraTimeoutError: If operations time out
        InfraUnavailableError: If circuit breaker is open
    """

    def __init__(
        self,
        timeout_query: ServiceTimeoutScanner,
        timeout_emission: ServiceTimeoutEmitter,
    ) -> None:
        """Initialize with required service dependencies.

        Args:
            timeout_query: Scanner for querying overdue entities.
            timeout_emission: Service for emitting timeout events.

        Example:
            >>> reader = ProjectionReaderRegistration(pool)
            >>> query = ServiceTimeoutScanner(container, reader)
            >>> emission = ServiceTimeoutEmitter(container, query, event_bus, projector)
            >>> coordinator = TimeoutCoordinator(query, emission)
        """
        self._timeout_query = timeout_query
        self._timeout_emission = timeout_emission

    async def coordinate(
        self,
        tick: ModelRuntimeTick,
        domain: str = "registration",
    ) -> ModelTimeoutCoordinationResult:
        """Coordinate a RuntimeTick event for timeout coordination.

        This is the main entry point for timeout coordination. It:
        1. Uses tick.now for all time-based decisions (injected time)
        2. Queries for overdue entities
        3. Emits timeout events and updates markers
        4. Returns comprehensive result for observability

        Args:
            tick: The RuntimeTick event with injected 'now'.
            domain: Domain namespace for queries (default: "registration").

        Returns:
            ModelTimeoutCoordinationResult with coordination details.

        Raises:
            InfraConnectionError: If database/Kafka connection fails
            InfraTimeoutError: If operations time out
            InfraUnavailableError: If circuit breaker is open

        Example:
            >>> tick = ModelRuntimeTick(
            ...     now=datetime.now(UTC),
            ...     tick_id=uuid4(),
            ...     sequence_number=1,
            ...     scheduled_at=datetime.now(UTC),
            ...     correlation_id=uuid4(),
            ...     scheduler_id="scheduler-001",
            ...     tick_interval_ms=1000,
            ... )
            >>> result = await coordinator.coordinate(tick)
            >>> print(f"Coordinated {result.total_found} timeout candidates")
        """
        start_time = time.perf_counter()

        # CRITICAL: Use tick.now, never system clock
        now = tick.now
        correlation_id = tick.correlation_id

        logger.debug(
            "Coordinating RuntimeTick for timeouts",
            extra={
                "tick_id": str(tick.tick_id),
                "now": now.isoformat(),
                "sequence_number": tick.sequence_number,
                "scheduler_id": tick.scheduler_id,
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # 1. Query for overdue entities
            query_start = time.perf_counter()
            query_result = await self._timeout_query.find_overdue_entities(
                now=now,
                domain=domain,
                correlation_id=correlation_id,
            )
            query_end = time.perf_counter()
            query_time_ms = (query_end - query_start) * 1000.0

            ack_timeouts_found = len(query_result.ack_timeouts)
            liveness_expirations_found = len(query_result.liveness_expirations)

            logger.debug(
                "Found overdue entities",
                extra={
                    "tick_id": str(tick.tick_id),
                    "ack_timeouts_found": ack_timeouts_found,
                    "liveness_expirations_found": liveness_expirations_found,
                    "query_time_ms": query_time_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            # 2. Emit timeout events (if any found)
            emission_start = time.perf_counter()
            emission_result = await self._timeout_emission.process_timeouts(
                now=now,
                tick_id=tick.tick_id,
                correlation_id=correlation_id,
                domain=domain,
            )
            emission_end = time.perf_counter()
            emission_time_ms = (emission_end - emission_start) * 1000.0

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000.0

            logger.info(
                "RuntimeTick timeout coordination completed",
                extra={
                    "tick_id": str(tick.tick_id),
                    "ack_timeouts_found": ack_timeouts_found,
                    "liveness_expirations_found": liveness_expirations_found,
                    "ack_timeouts_emitted": emission_result.ack_timeouts_emitted,
                    "liveness_expirations_emitted": emission_result.liveness_expirations_emitted,
                    "markers_updated": emission_result.markers_updated,
                    "coordination_time_ms": total_time_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelTimeoutCoordinationResult(
                tick_id=tick.tick_id,
                tick_now=now,
                ack_timeouts_found=ack_timeouts_found,
                liveness_expirations_found=liveness_expirations_found,
                ack_timeouts_emitted=emission_result.ack_timeouts_emitted,
                liveness_expirations_emitted=emission_result.liveness_expirations_emitted,
                markers_updated=emission_result.markers_updated,
                coordination_time_ms=total_time_ms,
                query_time_ms=query_time_ms,
                emission_time_ms=emission_time_ms,
                success=True,
                errors=emission_result.errors,
            )

        except Exception as e:
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000.0

            logger.exception(
                "RuntimeTick timeout coordination failed",
                extra={
                    "tick_id": str(tick.tick_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "coordination_time_ms": total_time_ms,
                    "correlation_id": str(correlation_id),
                },
            )

            return ModelTimeoutCoordinationResult(
                tick_id=tick.tick_id,
                tick_now=now,
                coordination_time_ms=total_time_ms,
                success=False,
                error=f"{type(e).__name__}: {e!s}",
            )


__all__: list[str] = ["ModelTimeoutCoordinationResult", "TimeoutCoordinator"]
