# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Writer for Agent Actions Observability.

This module provides a PostgreSQL writer for persisting agent observability
events consumed from Kafka. It handles batch inserts with idempotency
guarantees and circuit breaker resilience.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed
    - Batch inserts: Uses executemany for efficient batch processing
    - Idempotency: ON CONFLICT DO NOTHING/UPDATE per table contract
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience
    - JSONB serialization: dict fields serialized to JSON strings

Idempotency Contract:
    | Table                         | Unique Key       | Conflict Action |
    |-------------------------------|------------------|-----------------|
    | agent_actions                 | id               | DO NOTHING      |
    | agent_routing_decisions       | id               | DO NOTHING      |
    | agent_transformation_events   | id               | DO NOTHING      |
    | router_performance_metrics    | id               | DO NOTHING      |
    | agent_detection_failures      | correlation_id   | DO NOTHING      |
    | agent_execution_logs          | execution_id     | DO UPDATE       |

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.agent_actions.writer_postgres import (
    ...     WriterAgentActionsPostgres,
    ... )
    >>>
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> writer = WriterAgentActionsPostgres(pool)
    >>>
    >>> # Write batch of agent actions
    >>> count = await writer.write_agent_actions(actions)
    >>> print(f"Wrote {count} agent actions")
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from uuid import UUID, uuid4

import asyncpg

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.services.observability.agent_actions.models import (
    ModelAgentAction,
    ModelDetectionFailure,
    ModelExecutionLog,
    ModelPerformanceMetric,
    ModelRoutingDecision,
    ModelTransformationEvent,
)

logger = logging.getLogger(__name__)


class WriterAgentActionsPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for agent observability events.

    Provides batch write methods for each observability table with idempotency
    guarantees and circuit breaker resilience. The asyncpg.Pool is injected
    and its lifecycle is managed externally.

    Features:
        - Batch inserts via executemany for efficiency
        - Idempotent writes via ON CONFLICT clauses
        - Circuit breaker for database resilience
        - JSONB field serialization
        - Correlation ID propagation for tracing

    Attributes:
        _pool: Injected asyncpg connection pool.
        circuit_breaker_threshold: Failure threshold before opening circuit.
        circuit_breaker_reset_timeout: Seconds before auto-reset.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> writer = WriterAgentActionsPostgres(
        ...     pool,
        ...     circuit_breaker_threshold=5,
        ...     circuit_breaker_reset_timeout=60.0,
        ...     circuit_breaker_half_open_successes=2,
        ... )
        >>>
        >>> # Write batch of routing decisions
        >>> count = await writer.write_routing_decisions(decisions)
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        circuit_breaker_half_open_successes: int = 1,
    ) -> None:
        """Initialize the PostgreSQL writer with an injected pool.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            circuit_breaker_threshold: Failures before opening circuit (default: 5).
            circuit_breaker_reset_timeout: Seconds before auto-reset (default: 60.0).
            circuit_breaker_half_open_successes: Successful requests required to close
                circuit from half-open state (default: 1).

        Raises:
            ProtocolConfigurationError: If circuit breaker parameters are invalid.
        """
        self._pool = pool

        # Initialize circuit breaker mixin
        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="agent-actions-postgres-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

        logger.info(
            "WriterAgentActionsPostgres initialized",
            extra={
                "circuit_breaker_threshold": circuit_breaker_threshold,
                "circuit_breaker_reset_timeout": circuit_breaker_reset_timeout,
                "circuit_breaker_half_open_successes": circuit_breaker_half_open_successes,
            },
        )

    @staticmethod
    def _serialize_json(value: Mapping[str, object] | None) -> str | None:
        """Serialize a mapping to JSON string for JSONB columns.

        Args:
            value: Mapping to serialize, or None.

        Returns:
            JSON string if value is not None, otherwise None.
        """
        if value is None:
            return None
        return json.dumps(dict(value))

    @staticmethod
    def _serialize_list(value: list[str] | None) -> str | None:
        """Serialize a list to JSON string for JSONB/array columns.

        Args:
            value: List to serialize, or None.

        Returns:
            JSON string if value is not None, otherwise None.
        """
        if value is None:
            return None
        return json.dumps(value)

    async def write_agent_actions(
        self,
        events: list[ModelAgentAction],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of agent action events to PostgreSQL.

        Uses INSERT ... ON CONFLICT (id) DO NOTHING for idempotency.
        Append-only audit log - duplicates are silently ignored.

        Args:
            events: List of agent action events to write.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch (executemany doesn't return affected rows).

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_agent_actions",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_agent_actions",
            target_name="agent_actions",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO agent_actions (
                id, correlation_id, agent_name, action_type, action_name,
                created_at, status, duration_ms, result, error_message, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO NOTHING
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e.id,
                            e.correlation_id,
                            e.agent_name,
                            e.action_type,
                            e.action_name,
                            e.created_at,
                            e.status,
                            e.duration_ms,
                            e.result,
                            e.error_message,
                            self._serialize_json(e.metadata),
                        )
                        for e in events
                    ],
                )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote agent actions batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_agent_actions",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write agent actions timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=30.0,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_agent_actions",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_agent_actions",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_agent_actions",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_agent_actions: {type(e).__name__}",
                context=context,
            ) from e

    async def write_routing_decisions(
        self,
        events: list[ModelRoutingDecision],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of routing decision events to PostgreSQL.

        Uses INSERT ... ON CONFLICT (id) DO NOTHING for idempotency.
        Append-only audit log - duplicates are silently ignored.

        Args:
            events: List of routing decision events to write.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_routing_decisions",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_routing_decisions",
            target_name="agent_routing_decisions",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO agent_routing_decisions (
                id, correlation_id, selected_agent, confidence_score, created_at,
                request_type, alternatives, routing_reason, domain, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO NOTHING
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e.id,
                            e.correlation_id,
                            e.selected_agent,
                            e.confidence_score,
                            e.created_at,
                            e.request_type,
                            self._serialize_list(e.alternatives),
                            e.routing_reason,
                            e.domain,
                            self._serialize_json(e.metadata),
                        )
                        for e in events
                    ],
                )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote routing decisions batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_routing_decisions",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write routing decisions timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=30.0,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_routing_decisions",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_routing_decisions",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_routing_decisions",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_routing_decisions: {type(e).__name__}",
                context=context,
            ) from e

    async def write_transformation_events(
        self,
        events: list[ModelTransformationEvent],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of transformation events to PostgreSQL.

        Uses INSERT ... ON CONFLICT (id) DO NOTHING for idempotency.
        Append-only audit log - duplicates are silently ignored.

        Args:
            events: List of transformation events to write.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_transformation_events",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_transformation_events",
            target_name="agent_transformation_events",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO agent_transformation_events (
                id, correlation_id, source_agent, target_agent, created_at,
                trigger, context, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO NOTHING
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e.id,
                            e.correlation_id,
                            e.source_agent,
                            e.target_agent,
                            e.created_at,
                            e.trigger,
                            e.context,
                            self._serialize_json(e.metadata),
                        )
                        for e in events
                    ],
                )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote transformation events batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_transformation_events",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write transformation events timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=30.0,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_transformation_events",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_transformation_events",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_transformation_events",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_transformation_events: {type(e).__name__}",
                context=context,
            ) from e

    async def write_performance_metrics(
        self,
        events: list[ModelPerformanceMetric],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of performance metrics to PostgreSQL.

        Uses INSERT ... ON CONFLICT (id) DO NOTHING for idempotency.
        Append-only time-series - duplicates are silently ignored.

        Args:
            events: List of performance metric events to write.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_performance_metrics",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_performance_metrics",
            target_name="router_performance_metrics",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO router_performance_metrics (
                id, metric_name, metric_value, created_at,
                correlation_id, unit, agent_name, labels, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (id) DO NOTHING
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e.id,
                            e.metric_name,
                            e.metric_value,
                            e.created_at,
                            e.correlation_id,
                            e.unit,
                            e.agent_name,
                            self._serialize_json(e.labels),
                            self._serialize_json(e.metadata),
                        )
                        for e in events
                    ],
                )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote performance metrics batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_performance_metrics",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write performance metrics timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=30.0,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_performance_metrics",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_performance_metrics",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_performance_metrics",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_performance_metrics: {type(e).__name__}",
                context=context,
            ) from e

    async def write_detection_failures(
        self,
        events: list[ModelDetectionFailure],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of detection failure events to PostgreSQL.

        Uses INSERT ... ON CONFLICT (correlation_id) DO NOTHING for idempotency.
        One failure per correlation - duplicates are silently ignored.

        Args:
            events: List of detection failure events to write.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_detection_failures",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_detection_failures",
            target_name="agent_detection_failures",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO agent_detection_failures (
                correlation_id, failure_reason, created_at,
                request_summary, attempted_patterns, fallback_used, error_code, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (correlation_id) DO NOTHING
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e.correlation_id,
                            e.failure_reason,
                            e.created_at,
                            e.request_summary,
                            self._serialize_list(e.attempted_patterns),
                            e.fallback_used,
                            e.error_code,
                            self._serialize_json(e.metadata),
                        )
                        for e in events
                    ],
                )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote detection failures batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_detection_failures",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write detection failures timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=30.0,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_detection_failures",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_detection_failures",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_detection_failures",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_detection_failures: {type(e).__name__}",
                context=context,
            ) from e

    async def write_execution_logs(
        self,
        events: list[ModelExecutionLog],
        correlation_id: UUID | None = None,
    ) -> int:
        """Write batch of execution log events to PostgreSQL.

        Uses INSERT ... ON CONFLICT (execution_id) DO UPDATE for lifecycle tracking.
        Supports status transitions (started -> running -> completed/failed).
        Updates status, completed_at, duration_ms, quality_score, error_message,
        error_type, metadata, and updated_at on conflict.

        Args:
            events: List of execution log events to write.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        op_correlation_id = correlation_id or uuid4()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_execution_logs",
                correlation_id=op_correlation_id,
            )

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="write_execution_logs",
            target_name="agent_execution_logs",
            correlation_id=op_correlation_id,
        )

        sql = """
            INSERT INTO agent_execution_logs (
                execution_id, correlation_id, agent_name, status,
                created_at, updated_at, started_at, completed_at,
                duration_ms, exit_code, error_message, input_summary,
                output_summary, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (execution_id) DO UPDATE SET
                status = EXCLUDED.status,
                completed_at = EXCLUDED.completed_at,
                duration_ms = EXCLUDED.duration_ms,
                exit_code = EXCLUDED.exit_code,
                error_message = EXCLUDED.error_message,
                output_summary = EXCLUDED.output_summary,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    sql,
                    [
                        (
                            e.execution_id,
                            e.correlation_id,
                            e.agent_name,
                            e.status,
                            e.created_at,
                            e.updated_at,
                            e.started_at,
                            e.completed_at,
                            e.duration_ms,
                            e.exit_code,
                            e.error_message,
                            e.input_summary,
                            e.output_summary,
                            self._serialize_json(e.metadata),
                        )
                        for e in events
                    ],
                )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote execution logs batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(op_correlation_id),
                },
            )
            return len(events)

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_execution_logs",
                    correlation_id=op_correlation_id,
                )
            raise InfraTimeoutError(
                "Write execution logs timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=context.transport_type,
                    operation=context.operation,
                    target_name=context.target_name,
                    correlation_id=context.correlation_id,
                    timeout_seconds=30.0,
                ),
            ) from e
        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_execution_logs",
                    correlation_id=op_correlation_id,
                )
            raise InfraConnectionError(
                "Database connection failed during write_execution_logs",
                context=context,
            ) from e
        except asyncpg.PostgresError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="write_execution_logs",
                    correlation_id=op_correlation_id,
                )
            raise RuntimeHostError(
                f"Database error during write_execution_logs: {type(e).__name__}",
                context=context,
            ) from e

    def get_circuit_breaker_state(self) -> dict[str, JsonType]:
        """Return current circuit breaker state for health checks.

        Returns:
            Dict containing circuit breaker state information.
        """
        return self._get_circuit_breaker_state()


__all__ = ["WriterAgentActionsPostgres"]
