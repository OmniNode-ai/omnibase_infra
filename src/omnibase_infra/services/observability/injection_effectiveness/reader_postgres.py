# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""PostgreSQL Reader for Injection Effectiveness Observability.

This module provides a PostgreSQL reader for querying injection effectiveness
data stored by WriterInjectionEffectivenessPostgres. Supports session lookup,
flexible filtering with pagination, latency breakdowns, and pattern hit rates.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed
    - Dynamic WHERE: Builds parameterized queries from non-None filter fields
    - Deterministic ordering: All queries use ORDER BY for reproducible results
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience
    - Positional parameters: Uses $1, $2, ... (asyncpg requirement)

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
    - OMN-1890: Store injection metrics with corrected schema

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.injection_effectiveness.reader_postgres import (
    ...     ReaderInjectionEffectivenessPostgres,
    ... )
    >>>
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> reader = ReaderInjectionEffectivenessPostgres(pool)
    >>>
    >>> # Query by session ID
    >>> row = await reader.query_by_session_id(session_id)
    >>> if row:
    ...     print(f"Utilization: {row.utilization_score}")
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_query import (
    ModelInjectionEffectivenessQuery,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_query_result import (
    ModelInjectionEffectivenessQueryResult,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_injection_effectiveness_row import (
    ModelInjectionEffectivenessRow,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_latency_breakdown_row import (
    ModelLatencyBreakdownRow,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_pattern_hit_rate_row import (
    ModelPatternHitRateRow,
)
from omnibase_infra.utils.util_db_error_context import db_operation_error_context

logger = logging.getLogger(__name__)


class ReaderInjectionEffectivenessPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL reader for injection effectiveness observability queries.

    Provides query methods for all three injection effectiveness tables with
    circuit breaker resilience and correlation ID propagation. The asyncpg.Pool
    is injected and its lifecycle is managed externally.

    Attributes:
        _pool: Injected asyncpg connection pool.
        DEFAULT_QUERY_TIMEOUT_SECONDS: Default timeout for queries (30s).
    """

    DEFAULT_QUERY_TIMEOUT_SECONDS: float = 30.0

    def __init__(
        self,
        pool: asyncpg.Pool,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        circuit_breaker_half_open_successes: int = 1,
        query_timeout: float | None = None,
    ) -> None:
        """Initialize the PostgreSQL reader with an injected pool.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            circuit_breaker_threshold: Failures before opening circuit (default: 5).
            circuit_breaker_reset_timeout: Seconds before auto-reset (default: 60.0).
            circuit_breaker_half_open_successes: Successes to close from half-open (default: 1).
            query_timeout: Timeout in seconds for queries (default: 30.0).
        """
        self._pool = pool
        self._query_timeout = (
            query_timeout
            if query_timeout is not None
            else self.DEFAULT_QUERY_TIMEOUT_SECONDS
        )

        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="injection-effectiveness-postgres-reader",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

        logger.info(
            "ReaderInjectionEffectivenessPostgres initialized",
            extra={
                "circuit_breaker_threshold": circuit_breaker_threshold,
                "query_timeout": self._query_timeout,
            },
        )

    async def query_by_session_id(
        self,
        session_id: UUID,
        correlation_id: UUID | None = None,
    ) -> ModelInjectionEffectivenessRow | None:
        """Query a single session's injection effectiveness data.

        Args:
            session_id: Session identifier (primary key).
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelInjectionEffectivenessRow if found, None otherwise.
        """
        if correlation_id is None:
            correlation_id = uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="query_by_session_id",
                correlation_id=correlation_id,
            )

        sql = """
            SELECT session_id, correlation_id, realm, runtime_id, routing_path,
                   cohort, cohort_identity_type, total_injected_tokens, patterns_injected,
                   utilization_score, utilization_method, injected_identifiers_count,
                   reused_identifiers_count, agent_match_score, expected_agent,
                   actual_agent, user_visible_latency_ms, created_at, updated_at
            FROM injection_effectiveness
            WHERE session_id = $1
        """

        async with db_operation_error_context(
            operation="query_by_session_id",
            target_name="injection_effectiveness",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                async with conn.transaction(readonly=True):
                    timeout_ms = int(self._query_timeout * 1000)
                    await conn.execute(
                        "SET LOCAL statement_timeout = $1", str(timeout_ms)
                    )

                    row = await conn.fetchrow(sql, session_id)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if row is None:
                return None

            return ModelInjectionEffectivenessRow(**dict(row))

    async def query(
        self,
        query: ModelInjectionEffectivenessQuery,
        correlation_id: UUID | None = None,
    ) -> ModelInjectionEffectivenessQueryResult:
        """Query with flexible filters, returns paginated results.

        Args:
            query: Query parameters with optional filters.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelInjectionEffectivenessQueryResult with pagination metadata.
        """
        # Defense-in-depth: mirrors Pydantic constraints on ModelInjectionEffectivenessQuery
        # (limit: ge=1, le=10000; offset: ge=0, le=1000000). Kept in sync manually.
        if query.limit < 1 or query.limit > 10000:
            msg = f"limit must be between 1 and 10000, got {query.limit}"
            raise ValueError(msg)
        if query.offset < 0 or query.offset > 1000000:
            msg = f"offset must be between 0 and 1000000, got {query.offset}"
            raise ValueError(msg)

        if correlation_id is None:
            correlation_id = uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="query",
                correlation_id=correlation_id,
            )

        # Build dynamic WHERE clause
        conditions: list[str] = []
        params: list[object] = []
        param_idx = 1

        if query.session_id is not None:
            conditions.append(f"session_id = ${param_idx}")
            params.append(query.session_id)
            param_idx += 1

        if query.correlation_id is not None:
            conditions.append(f"correlation_id = ${param_idx}")
            params.append(query.correlation_id)
            param_idx += 1

        if query.cohort is not None:
            conditions.append(f"cohort = ${param_idx}")
            params.append(query.cohort)
            param_idx += 1

        if query.utilization_method is not None:
            conditions.append(f"utilization_method = ${param_idx}")
            params.append(query.utilization_method)
            param_idx += 1

        if query.start_time is not None:
            conditions.append(f"created_at >= ${param_idx}")
            params.append(query.start_time)
            param_idx += 1

        if query.end_time is not None:
            conditions.append(f"created_at < ${param_idx}")
            params.append(query.end_time)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        # Count query for pagination metadata
        count_sql = f"SELECT COUNT(*) FROM injection_effectiveness WHERE {where_clause}"  # noqa: S608

        # Data query with pagination and deterministic ordering
        limit_idx = param_idx
        offset_idx = param_idx + 1
        param_idx = offset_idx + 1  # Keep idx consistent for future appends

        data_sql = f"""
            SELECT session_id, correlation_id, realm, runtime_id, routing_path,
                   cohort, cohort_identity_type, total_injected_tokens, patterns_injected,
                   utilization_score, utilization_method, injected_identifiers_count,
                   reused_identifiers_count, agent_match_score, expected_agent,
                   actual_agent, user_visible_latency_ms, created_at, updated_at
            FROM injection_effectiveness
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """  # noqa: S608

        data_params = [*params, query.limit, query.offset]

        async with db_operation_error_context(
            operation="query",
            target_name="injection_effectiveness",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                async with conn.transaction(readonly=True):
                    timeout_ms = int(self._query_timeout * 1000)
                    await conn.execute(
                        "SET LOCAL statement_timeout = $1", str(timeout_ms)
                    )

                    raw_count = await conn.fetchval(count_sql, *params)
                    total_count: int = raw_count if raw_count is not None else 0
                    rows = await conn.fetch(data_sql, *data_params)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            result_rows = tuple(ModelInjectionEffectivenessRow(**dict(r)) for r in rows)

            return ModelInjectionEffectivenessQueryResult(
                rows=result_rows,
                total_count=total_count,
                has_more=(query.offset + query.limit) < total_count,
                query=query,
            )

    async def query_latency_breakdowns(
        self,
        session_id: UUID,
        correlation_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelLatencyBreakdownRow]:
        """Query latency breakdowns for a specific session.

        Args:
            session_id: Session identifier.
            correlation_id: Correlation ID for tracing.
            limit: Maximum rows to return (1-10000).
            offset: Pagination offset (>= 0).

        Returns:
            List of ModelLatencyBreakdownRow ordered by created_at ASC.

        Raises:
            ValueError: If limit or offset is out of bounds.
        """
        if limit < 1 or limit > 10000:
            msg = f"limit must be between 1 and 10000, got {limit}"
            raise ValueError(msg)
        if offset < 0 or offset > 1000000:
            msg = f"offset must be between 0 and 1000000, got {offset}"
            raise ValueError(msg)

        if correlation_id is None:
            correlation_id = uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="query_latency_breakdowns",
                correlation_id=correlation_id,
            )

        sql = """
            SELECT id, session_id, prompt_id, cohort, cache_hit,
                   routing_latency_ms, retrieval_latency_ms, injection_latency_ms,
                   user_latency_ms, emitted_at, created_at
            FROM latency_breakdowns
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
        """

        async with db_operation_error_context(
            operation="query_latency_breakdowns",
            target_name="latency_breakdowns",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                async with conn.transaction(readonly=True):
                    timeout_ms = int(self._query_timeout * 1000)
                    await conn.execute(
                        "SET LOCAL statement_timeout = $1", str(timeout_ms)
                    )

                    rows = await conn.fetch(sql, session_id, limit, offset)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [ModelLatencyBreakdownRow(**dict(r)) for r in rows]

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
            pattern_id: Filter by pattern ID (None = all).
            confident_only: Only return patterns with confidence != NULL.
            correlation_id: Correlation ID for tracing.
            limit: Maximum rows to return (1-10000).
            offset: Pagination offset (>= 0).

        Returns:
            List of ModelPatternHitRateRow ordered by updated_at DESC.

        Raises:
            ValueError: If limit or offset is out of bounds.
        """
        if limit < 1 or limit > 10000:
            msg = f"limit must be between 1 and 10000, got {limit}"
            raise ValueError(msg)
        if offset < 0 or offset > 1000000:
            msg = f"offset must be between 0 and 1000000, got {offset}"
            raise ValueError(msg)

        if correlation_id is None:
            correlation_id = uuid4()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="query_pattern_hit_rates",
                correlation_id=correlation_id,
            )

        conditions: list[str] = []
        params: list[object] = []
        param_idx = 1

        if pattern_id is not None:
            conditions.append(f"pattern_id = ${param_idx}")
            params.append(pattern_id)
            param_idx += 1

        if confident_only:
            conditions.append("confidence IS NOT NULL")

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        limit_idx = param_idx
        offset_idx = param_idx + 1
        param_idx = offset_idx + 1  # Keep idx consistent for future appends

        sql = f"""
            SELECT id, pattern_id, domain_id, utilization_method, utilization_score,
                   hit_count, miss_count, sample_count, confidence,
                   created_at, updated_at
            FROM pattern_hit_rates
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """  # noqa: S608

        query_params = [*params, limit, offset]

        async with db_operation_error_context(
            operation="query_pattern_hit_rates",
            target_name="pattern_hit_rates",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                async with conn.transaction(readonly=True):
                    timeout_ms = int(self._query_timeout * 1000)
                    await conn.execute(
                        "SET LOCAL statement_timeout = $1", str(timeout_ms)
                    )

                    rows = await conn.fetch(sql, *query_params)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [ModelPatternHitRateRow(**dict(r)) for r in rows]


__all__ = ["ReaderInjectionEffectivenessPostgres"]
