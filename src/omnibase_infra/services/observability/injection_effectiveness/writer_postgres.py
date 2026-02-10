# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Writer for Injection Effectiveness Observability.

This module provides a PostgreSQL writer for persisting injection effectiveness
events consumed from Kafka. It handles batch upserts with idempotency
guarantees and circuit breaker resilience.

Design Decisions:
    - Pool injection: asyncpg.Pool is injected, not created/managed
    - Batch inserts: Uses executemany for efficient batch processing
    - Idempotency: ON CONFLICT DO NOTHING/UPDATE per table contract
    - Circuit breaker: MixinAsyncCircuitBreaker for resilience

Idempotency Contract:
    | Table                  | Unique Key                   | Conflict Action |
    |------------------------|------------------------------|-----------------|
    | injection_effectiveness| session_id                   | DO UPDATE       |
    | latency_breakdowns     | (session_id, prompt_id)      | DO NOTHING      |
    | pattern_hit_rates      | (pattern_id, utilization_method) | DO UPDATE (rolling avg) |

Related Tickets:
    - OMN-1890: Store injection metrics with corrected schema

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
    ...     WriterInjectionEffectivenessPostgres,
    ... )
    >>>
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> writer = WriterInjectionEffectivenessPostgres(pool)
    >>>
    >>> # Write batch of context utilization events
    >>> count = await writer.write_context_utilization(events)
    >>> print(f"Wrote {count} context utilization events")
"""

from __future__ import annotations

import logging
from uuid import UUID

import asyncpg

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.services.observability.injection_effectiveness.models.model_agent_match import (
    ModelAgentMatchEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_context_utilization import (
    ModelContextUtilizationEvent,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_latency_breakdown import (
    ModelLatencyBreakdownEvent,
)
from omnibase_infra.utils.util_db_error_context import db_operation_error_context

logger = logging.getLogger(__name__)


class WriterInjectionEffectivenessPostgres(MixinAsyncCircuitBreaker):
    """PostgreSQL writer for injection effectiveness observability events.

    Provides batch write methods for injection effectiveness tables with idempotency
    guarantees and circuit breaker resilience. The asyncpg.Pool is injected
    and its lifecycle is managed externally.

    Features:
        - Batch inserts/upserts via executemany for efficiency
        - Idempotent writes via ON CONFLICT clauses
        - Circuit breaker for database resilience
        - Correlation ID propagation for tracing

    Attributes:
        _pool: Injected asyncpg connection pool.
        circuit_breaker_threshold: Failure threshold before opening circuit.
        circuit_breaker_reset_timeout: Seconds before auto-reset.
        DEFAULT_QUERY_TIMEOUT_SECONDS: Default timeout for database queries (30s).
        DEFAULT_MINIMUM_SUPPORT_THRESHOLD: Default minimum sample count for confidence (20).
        DEFAULT_HIT_MISS_THRESHOLD: Default threshold for hit/miss classification (0.5).

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> writer = WriterInjectionEffectivenessPostgres(
        ...     pool,
        ...     circuit_breaker_threshold=5,
        ...     circuit_breaker_reset_timeout=60.0,
        ...     circuit_breaker_half_open_successes=2,
        ...     query_timeout=30.0,
        ...     minimum_support_threshold=20,  # samples needed before confidence
        ...     hit_miss_threshold=0.5,  # score threshold for hit vs miss
        ... )
        >>>
        >>> # Write batch of context utilization events
        >>> count = await writer.write_context_utilization(events)
    """

    DEFAULT_QUERY_TIMEOUT_SECONDS: float = 30.0
    DEFAULT_MINIMUM_SUPPORT_THRESHOLD: int = 20
    DEFAULT_HIT_MISS_THRESHOLD: float = 0.5

    def __init__(
        self,
        pool: asyncpg.Pool,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
        circuit_breaker_half_open_successes: int = 1,
        query_timeout: float | None = None,
        minimum_support_threshold: int | None = None,
        hit_miss_threshold: float | None = None,
    ) -> None:
        """Initialize the PostgreSQL writer with an injected pool.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            circuit_breaker_threshold: Failures before opening circuit (default: 5).
            circuit_breaker_reset_timeout: Seconds before auto-reset (default: 60.0).
            circuit_breaker_half_open_successes: Successful requests required to close
                circuit from half-open state (default: 1).
            query_timeout: Timeout in seconds for database queries. Applied via
                PostgreSQL statement_timeout (default: DEFAULT_QUERY_TIMEOUT_SECONDS).
            minimum_support_threshold: Minimum sample count required before calculating
                confidence score for pattern_hit_rates. This implements statistical
                minimum support gating to avoid premature confidence scores based on
                insufficient data (default: DEFAULT_MINIMUM_SUPPORT_THRESHOLD = 20).
            hit_miss_threshold: Threshold for classifying pattern utilization as hit
                vs miss. Scores > threshold count as hits, scores <= threshold count
                as misses. This heuristic determines when a pattern injection was
                "useful enough" to count as a hit (default: DEFAULT_HIT_MISS_THRESHOLD = 0.5).

        Raises:
            ProtocolConfigurationError: If circuit breaker parameters are invalid.
        """
        self._pool = pool
        self._query_timeout = (
            query_timeout
            if query_timeout is not None
            else self.DEFAULT_QUERY_TIMEOUT_SECONDS
        )
        self._minimum_support_threshold = (
            minimum_support_threshold
            if minimum_support_threshold is not None
            else self.DEFAULT_MINIMUM_SUPPORT_THRESHOLD
        )
        self._hit_miss_threshold = (
            hit_miss_threshold
            if hit_miss_threshold is not None
            else self.DEFAULT_HIT_MISS_THRESHOLD
        )

        # Initialize circuit breaker mixin
        self._init_circuit_breaker(
            threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout,
            service_name="injection-effectiveness-postgres-writer",
            transport_type=EnumInfraTransportType.DATABASE,
            half_open_successes=circuit_breaker_half_open_successes,
        )

        logger.info(
            "WriterInjectionEffectivenessPostgres initialized",
            extra={
                "circuit_breaker_threshold": circuit_breaker_threshold,
                "circuit_breaker_reset_timeout": circuit_breaker_reset_timeout,
                "circuit_breaker_half_open_successes": circuit_breaker_half_open_successes,
                "query_timeout": self._query_timeout,
                "minimum_support_threshold": self._minimum_support_threshold,
                "hit_miss_threshold": self._hit_miss_threshold,
            },
        )

    async def write_context_utilization(
        self,
        events: list[ModelContextUtilizationEvent],
        correlation_id: UUID,
    ) -> int:
        """Write batch of context utilization events to PostgreSQL.

        Performs two operations:
            1. UPSERT to injection_effectiveness table (session_id is primary key)
            2. INSERT to pattern_hit_rates table for each pattern (ON CONFLICT DO NOTHING)

        Args:
            events: List of context utilization events to write.
            correlation_id: Correlation ID for tracing (required - models auto-generate).

        Returns:
            Count of events in the batch (executemany doesn't return affected rows).

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        # Check circuit breaker before entering error context
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_context_utilization",
                correlation_id=correlation_id,
            )

        # SQL for injection_effectiveness upsert
        sql_effectiveness = """
            INSERT INTO injection_effectiveness (
                session_id, correlation_id, cohort, cohort_identity_type,
                total_injected_tokens, patterns_injected, utilization_score,
                utilization_method, injected_identifiers_count, reused_identifiers_count,
                created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                correlation_id = COALESCE(EXCLUDED.correlation_id, injection_effectiveness.correlation_id),
                cohort = COALESCE(EXCLUDED.cohort, injection_effectiveness.cohort),
                cohort_identity_type = COALESCE(EXCLUDED.cohort_identity_type, injection_effectiveness.cohort_identity_type),
                total_injected_tokens = EXCLUDED.total_injected_tokens,
                patterns_injected = EXCLUDED.patterns_injected,
                utilization_score = EXCLUDED.utilization_score,
                utilization_method = EXCLUDED.utilization_method,
                injected_identifiers_count = EXCLUDED.injected_identifiers_count,
                reused_identifiers_count = EXCLUDED.reused_identifiers_count,
                updated_at = NOW()
        """

        # SQL for pattern_hit_rates upsert with rolling average
        # Aggregates per-pattern statistics across all sessions
        # Note: minimum_support_threshold is formatted into SQL since executemany
        # doesn't support different parameter values per-position, and this is a
        # controlled integer configuration value (not user input).
        # Security: int() cast guarantees numeric-only output, preventing SQL injection.
        min_support_str = str(int(self._minimum_support_threshold))
        sql_patterns = """
            INSERT INTO pattern_hit_rates (
                pattern_id, utilization_method, utilization_score,
                hit_count, miss_count, sample_count, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, 1, NOW(), NOW())
            ON CONFLICT (pattern_id, utilization_method) DO UPDATE SET
                -- Rolling average: new_avg = ((old_avg * old_count) + new_score) / (old_count + 1)
                utilization_score = (
                    (pattern_hit_rates.utilization_score * pattern_hit_rates.sample_count) + EXCLUDED.utilization_score
                ) / (pattern_hit_rates.sample_count + 1),
                hit_count = pattern_hit_rates.hit_count + EXCLUDED.hit_count,
                miss_count = pattern_hit_rates.miss_count + EXCLUDED.miss_count,
                sample_count = pattern_hit_rates.sample_count + 1,
                -- Set confidence when sample_count >= minimum_support_threshold
                -- (minimum support gating prevents premature confidence scores)
                confidence = CASE
                    WHEN pattern_hit_rates.sample_count + 1 >= __MIN_SUPPORT__ THEN
                        (pattern_hit_rates.utilization_score * pattern_hit_rates.sample_count + EXCLUDED.utilization_score) / (pattern_hit_rates.sample_count + 1)
                    ELSE NULL
                END,
                updated_at = NOW()
        """.replace("__MIN_SUPPORT__", min_support_str)

        # Use shared error context for consistent exception handling
        async with db_operation_error_context(
            operation="write_context_utilization",
            target_name="injection_effectiveness",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                # Apply statement_timeout for query timeout enforcement
                # Convert seconds to milliseconds for PostgreSQL
                # Use parameterized query for defense in depth (even though int() cast
                # already guarantees numeric output, parameterized is the preferred pattern)
                timeout_ms = int(self._query_timeout * 1000)
                await conn.execute("SET statement_timeout = $1", str(timeout_ms))

                # Wrap both writes in an explicit transaction for atomicity.
                # If pattern_hit_rates write fails after injection_effectiveness succeeds,
                # both are rolled back to prevent partial data.
                async with conn.transaction():
                    # Write to injection_effectiveness
                    await conn.executemany(
                        sql_effectiveness,
                        [
                            (
                                e.session_id,
                                e.correlation_id,
                                e.cohort,
                                e.cohort_identity_type,
                                e.total_injected_tokens,
                                e.patterns_injected,
                                e.utilization_score,
                                e.utilization_method,
                                e.injected_identifiers_count,
                                e.reused_identifiers_count,
                                e.created_at,
                            )
                            for e in events
                        ],
                    )

                    # Write pattern utilizations to pattern_hit_rates (aggregated per pattern)
                    #
                    # Hit/miss classification threshold rationale:
                    # The default threshold of 0.5 represents a "majority utility" heuristic:
                    # - hit (score > 0.5): More than half the injected pattern content was
                    #   utilized by the model, indicating the injection was net-positive.
                    # - miss (score <= 0.5): Half or less was utilized, indicating the
                    #   injection added noise/tokens without proportional benefit.
                    #
                    # This threshold is configurable via hit_miss_threshold parameter to
                    # accommodate different utilization measurement methods and use cases.
                    # For example, strict environments might use 0.7, while exploratory
                    # injections might tolerate 0.3.
                    #
                    # Classification is binary (hit=1/miss=1) to enable simple aggregate
                    # hit rate calculations: hit_rate = hit_count / (hit_count + miss_count)
                    pattern_rows = []
                    for e in events:
                        for p in e.pattern_utilizations:
                            hit_count = (
                                1
                                if p.utilization_score > self._hit_miss_threshold
                                else 0
                            )
                            miss_count = (
                                0
                                if p.utilization_score > self._hit_miss_threshold
                                else 1
                            )
                            pattern_rows.append(
                                (
                                    p.pattern_id,
                                    p.utilization_method,
                                    p.utilization_score,
                                    hit_count,
                                    miss_count,
                                )
                            )

                    if pattern_rows:
                        await conn.executemany(sql_patterns, pattern_rows)

            # Record success - reset circuit breaker after successful write
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote context utilization batch",
                extra={
                    "count": len(events),
                    "pattern_count": len(pattern_rows) if pattern_rows else 0,
                    "correlation_id": str(correlation_id),
                },
            )
            return len(events)

    async def write_agent_match(
        self,
        events: list[ModelAgentMatchEvent],
        correlation_id: UUID,
    ) -> int:
        """Write batch of agent match events to PostgreSQL.

        UPSERT to injection_effectiveness table, merging with existing session data.
        Only updates agent match fields (agent_match_score, expected_agent, actual_agent).

        Args:
            events: List of agent match events to write.
            correlation_id: Correlation ID for tracing (required - models auto-generate).

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        # Check circuit breaker before entering error context
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_agent_match",
                correlation_id=correlation_id,
            )

        sql = """
            INSERT INTO injection_effectiveness (
                session_id, correlation_id, agent_match_score, expected_agent,
                actual_agent, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                correlation_id = COALESCE(EXCLUDED.correlation_id, injection_effectiveness.correlation_id),
                agent_match_score = EXCLUDED.agent_match_score,
                expected_agent = EXCLUDED.expected_agent,
                actual_agent = EXCLUDED.actual_agent,
                updated_at = NOW()
        """

        # Use shared error context for consistent exception handling
        async with db_operation_error_context(
            operation="write_agent_match",
            target_name="injection_effectiveness",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                # Apply statement_timeout for query timeout enforcement
                # Use parameterized query for defense in depth
                timeout_ms = int(self._query_timeout * 1000)
                await conn.execute("SET statement_timeout = $1", str(timeout_ms))

                await conn.executemany(
                    sql,
                    [
                        (
                            e.session_id,
                            e.correlation_id,
                            e.agent_match_score,
                            e.expected_agent,
                            e.actual_agent,
                            e.created_at,
                        )
                        for e in events
                    ],
                )

            # Record success - reset circuit breaker after successful write
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote agent match batch",
                extra={
                    "count": len(events),
                    "correlation_id": str(correlation_id),
                },
            )
            return len(events)

    async def write_latency_breakdowns(
        self,
        events: list[ModelLatencyBreakdownEvent],
        correlation_id: UUID,
    ) -> int:
        """Write batch of latency breakdown events to PostgreSQL.

        Performs two operations (order matters for FK constraint):
            1. UPSERT to injection_effectiveness table (creates parent row if needed)
            2. INSERT to latency_breakdowns table (ON CONFLICT DO NOTHING)

        Args:
            events: List of latency breakdown events to write.
            correlation_id: Correlation ID for tracing (required - models auto-generate).

        Returns:
            Count of events in the batch.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        if not events:
            return 0

        # Check circuit breaker before entering error context
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="write_latency_breakdowns",
                correlation_id=correlation_id,
            )

        # SQL for latency_breakdowns insert
        sql_breakdowns = """
            INSERT INTO latency_breakdowns (
                session_id, prompt_id, cohort, cache_hit,
                routing_latency_ms, retrieval_latency_ms, injection_latency_ms,
                user_latency_ms, emitted_at, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            ON CONFLICT (session_id, prompt_id) DO NOTHING
        """

        # SQL for injection_effectiveness upsert (MAX aggregation for user_visible_latency_ms)
        sql_effectiveness = """
            INSERT INTO injection_effectiveness (
                session_id, correlation_id, cohort, user_visible_latency_ms,
                created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                correlation_id = COALESCE(EXCLUDED.correlation_id, injection_effectiveness.correlation_id),
                cohort = COALESCE(EXCLUDED.cohort, injection_effectiveness.cohort),
                user_visible_latency_ms = GREATEST(
                    COALESCE(injection_effectiveness.user_visible_latency_ms, 0),
                    EXCLUDED.user_visible_latency_ms
                ),
                updated_at = NOW()
        """

        # Use shared error context for consistent exception handling
        async with db_operation_error_context(
            operation="write_latency_breakdowns",
            target_name="latency_breakdowns",
            correlation_id=correlation_id,
            timeout_seconds=self._query_timeout,
            circuit_breaker=self,
        ):
            async with self._pool.acquire() as conn:
                # Apply statement_timeout for query timeout enforcement
                # Use parameterized query for defense in depth
                timeout_ms = int(self._query_timeout * 1000)
                await conn.execute("SET statement_timeout = $1", str(timeout_ms))

                # IMPORTANT: Upsert to injection_effectiveness FIRST to satisfy FK constraint
                # If latency event arrives before utilization/agent-match events, we need
                # the parent row to exist before inserting the child row.

                # Compute MAX user_latency_ms per session for the batch
                session_latencies: dict[
                    UUID, tuple[int, ModelLatencyBreakdownEvent]
                ] = {}
                for e in events:
                    if e.session_id not in session_latencies:
                        session_latencies[e.session_id] = (e.user_latency_ms, e)
                    else:
                        existing_latency, _ = session_latencies[e.session_id]
                        if e.user_latency_ms > existing_latency:
                            session_latencies[e.session_id] = (e.user_latency_ms, e)

                # Wrap both writes in an explicit transaction for atomicity.
                # If latency_breakdowns insert fails after injection_effectiveness upsert,
                # both are rolled back to prevent partial data.
                async with conn.transaction():
                    # 1. First: Upsert to injection_effectiveness (creates parent row if needed)
                    await conn.executemany(
                        sql_effectiveness,
                        [
                            (
                                session_id,
                                event.correlation_id,
                                event.cohort,
                                max_latency,
                                event.created_at,
                            )
                            for session_id, (
                                max_latency,
                                event,
                            ) in session_latencies.items()
                        ],
                    )

                    # 2. Then: Insert to latency_breakdowns (FK now satisfied)
                    await conn.executemany(
                        sql_breakdowns,
                        [
                            (
                                e.session_id,
                                e.prompt_id,
                                e.cohort,
                                e.cache_hit,
                                e.routing_latency_ms,
                                e.retrieval_latency_ms,
                                e.injection_latency_ms,
                                e.user_latency_ms,
                                e.emitted_at,
                            )
                            for e in events
                        ],
                    )

            # Record success - reset circuit breaker after successful write
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Wrote latency breakdowns batch",
                extra={
                    "count": len(events),
                    "sessions_updated": len(session_latencies),
                    "correlation_id": str(correlation_id),
                },
            )
            return len(events)

    def get_circuit_breaker_state(self) -> dict[str, JsonType]:
        """Return current circuit breaker state for health checks.

        Returns:
            Dict containing circuit breaker state information.
        """
        return self._get_circuit_breaker_state()


__all__ = ["WriterInjectionEffectivenessPostgres"]
