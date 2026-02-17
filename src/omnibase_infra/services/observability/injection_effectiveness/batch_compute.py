# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Batch computation for effectiveness metrics from existing observability data.

Derives injection effectiveness metrics from the existing ``agent_actions``
and ``agent_routing_decisions`` tables, populating the three effectiveness
measurement tables (``injection_effectiveness``, ``latency_breakdowns``,
``pattern_hit_rates``).

This module bridges the gap when the Kafka pipeline is not yet producing
events: it computes effectiveness metrics from data that already exists
in the database, seeding the effectiveness tables with real measurements.

Design Decisions:
    - Read from agent_actions + agent_routing_decisions (already populated)
    - Write to injection_effectiveness, latency_breakdowns, pattern_hit_rates
    - Idempotent: Uses ON CONFLICT to avoid duplicates on re-runs
    - Batched: Processes in configurable batch sizes for memory efficiency
    - Pool injection: asyncpg.Pool injected, lifecycle managed externally
    - Correlation-aware: Joins on correlation_id to link actions to routing

Metrics Derivation Logic:
    - **injection_effectiveness**: One row per unique correlation_id from
      agent_routing_decisions. Utilization derived from action success rates.
      Agent match fields are NULL until expected-agent tracking is available.
    - **latency_breakdowns**: One row per agent_action with total duration_ms.
      Sub-component latencies (routing, retrieval, injection) are NULL
      because individual timing is not yet instrumented.
    - **pattern_hit_rates**: Aggregated from routing decisions grouped by
      selected_agent, treating agent selection as the "pattern".

Related Tickets:
    - OMN-2303: Activate effectiveness consumer and populate measurement tables

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.injection_effectiveness.batch_compute import (
    ...     BatchComputeEffectivenessMetrics,
    ... )
    >>>
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> batch = BatchComputeEffectivenessMetrics(pool)
    >>> result = await batch.compute_and_persist()
    >>> print(f"Wrote {result.effectiveness_rows} effectiveness rows")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.services.observability.injection_effectiveness.notifier import (
    EffectivenessInvalidationNotifier,
)
from omnibase_infra.utils.util_db_transaction import set_statement_timeout

logger = logging.getLogger(__name__)

# Default batch size for processing routing decisions
DEFAULT_BATCH_SIZE: int = 500

# Default query timeout in seconds
DEFAULT_QUERY_TIMEOUT: float = 60.0


@dataclass(frozen=True)
class BatchComputeResult:
    """Result of a batch computation run.

    Attributes:
        effectiveness_rows: Rows written to injection_effectiveness.
        latency_rows: Rows written to latency_breakdowns.
        pattern_rows: Rows written to pattern_hit_rates.
        errors: Error messages for any failed batches.
        started_at: Computation start timestamp.
        completed_at: Computation end timestamp.
    """

    effectiveness_rows: int = 0
    latency_rows: int = 0
    pattern_rows: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_rows(self) -> int:
        """Total rows written across all tables."""
        return self.effectiveness_rows + self.latency_rows + self.pattern_rows

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred during computation."""
        return len(self.errors) > 0


class BatchComputeEffectivenessMetrics:
    """Batch computation engine for effectiveness metrics.

    Reads existing data from agent_actions and agent_routing_decisions
    tables and derives effectiveness metrics for the three measurement
    tables.

    The computation is idempotent: running it multiple times produces
    the same result due to ON CONFLICT handling in all INSERT statements.

    Attributes:
        _pool: Injected asyncpg connection pool.
        _batch_size: Number of routing decisions to process per batch.
        _query_timeout: Query timeout in seconds.
        _notifier: Optional notifier for invalidation events.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> batch = BatchComputeEffectivenessMetrics(pool, batch_size=200)
        >>> result = await batch.compute_and_persist()
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
        notifier: EffectivenessInvalidationNotifier | None = None,
    ) -> None:
        """Initialize batch computation engine.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            batch_size: Routing decisions to process per batch.
            query_timeout: Query timeout in seconds.
            notifier: Optional notifier for publishing invalidation events
                after successful writes.
        """
        self._pool = pool
        self._batch_size = batch_size
        self._query_timeout = query_timeout
        self._notifier = notifier
        self._has_uuid_ossp: bool | None = None

    async def compute_and_persist(
        self,
        correlation_id: UUID | None = None,
    ) -> BatchComputeResult:
        """Run the full batch computation pipeline.

        Executes three computation phases:
            1. Derive injection_effectiveness rows from routing decisions
            2. Derive latency_breakdowns from agent action durations
            3. Derive pattern_hit_rates from agent selection patterns

        All writes are idempotent (ON CONFLICT DO NOTHING / DO UPDATE).

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            BatchComputeResult with counts and any errors.
        """
        effective_correlation_id = correlation_id or uuid4()
        started_at = datetime.now(UTC)
        errors: list[str] = []

        logger.info(
            "Starting batch effectiveness computation",
            extra={
                "correlation_id": str(effective_correlation_id),
                "batch_size": self._batch_size,
            },
        )

        # Phase 1: injection_effectiveness from routing decisions
        effectiveness_rows = 0
        try:
            effectiveness_rows = await self._compute_effectiveness(
                effective_correlation_id
            )
        except Exception as e:
            msg = f"Phase 1 (injection_effectiveness) failed: {e}"
            logger.exception(
                msg, extra={"correlation_id": str(effective_correlation_id)}
            )
            errors.append(msg)

        # Phase 2: latency_breakdowns from agent action durations
        latency_rows = 0
        try:
            latency_rows = await self._compute_latency_breakdowns(
                effective_correlation_id
            )
        except Exception as e:
            msg = f"Phase 2 (latency_breakdowns) failed: {e}"
            logger.exception(
                msg, extra={"correlation_id": str(effective_correlation_id)}
            )
            errors.append(msg)

        # Phase 3: pattern_hit_rates from agent selection patterns
        pattern_rows = 0
        try:
            pattern_rows = await self._compute_pattern_hit_rates(
                effective_correlation_id
            )
        except Exception as e:
            msg = f"Phase 3 (pattern_hit_rates) failed: {e}"
            logger.exception(
                msg, extra={"correlation_id": str(effective_correlation_id)}
            )
            errors.append(msg)

        completed_at = datetime.now(UTC)

        result = BatchComputeResult(
            effectiveness_rows=effectiveness_rows,
            latency_rows=latency_rows,
            pattern_rows=pattern_rows,
            errors=tuple(errors),
            started_at=started_at,
            completed_at=completed_at,
        )

        logger.info(
            "Batch effectiveness computation completed",
            extra={
                "correlation_id": str(effective_correlation_id),
                "effectiveness_rows": effectiveness_rows,
                "latency_rows": latency_rows,
                "pattern_rows": pattern_rows,
                "total_rows": result.total_rows,
                "has_errors": result.has_errors,
                "duration_seconds": (completed_at - started_at).total_seconds(),
            },
        )

        # Emit invalidation event if rows were written
        if result.total_rows > 0 and self._notifier is not None:
            tables: list[str] = []
            if effectiveness_rows > 0:
                tables.append("injection_effectiveness")
            if latency_rows > 0:
                tables.append("latency_breakdowns")
            if pattern_rows > 0:
                tables.append("pattern_hit_rates")

            await self._notifier.notify(
                tables_affected=tuple(tables),
                rows_written=result.total_rows,
                source="batch_compute",
                correlation_id=effective_correlation_id,
            )

        return result

    async def _compute_effectiveness(self, correlation_id: UUID) -> int:
        """Derive injection_effectiveness rows from routing decisions.

        Each unique correlation_id in agent_routing_decisions becomes one
        row in injection_effectiveness. The utilization_score is derived
        from the action success rate for that correlation. Agent match
        fields (expected_agent, agent_match_score) are NULL until
        expected-agent tracking is implemented.

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows written.
        """
        # This query:
        # 1. Iterates agent_routing_decisions rows, deduplicating by session_id
        #    via ON CONFLICT DO NOTHING (no GROUP BY; one row per routing decision)
        # 2. LATERAL JOINs with agent_actions to compute action success rates
        # 3. Derives utilization_score from completed/total action ratio
        # 4. Sets agent_match_score and expected_agent to NULL (not yet tracked)
        # 5. Computes user_visible_latency_ms from MAX(duration_ms)
        sql = """
            INSERT INTO injection_effectiveness (
                session_id, correlation_id, cohort,
                utilization_score, utilization_method,
                agent_match_score, expected_agent, actual_agent,
                user_visible_latency_ms,
                created_at, updated_at
            )
            SELECT
                rd.correlation_id AS session_id,
                rd.correlation_id,
                CASE
                    WHEN rd.confidence_score >= 0.8 THEN 'treatment'
                    ELSE 'control'
                END AS cohort,
                -- utilization_score: ratio of completed actions to total actions
                COALESCE(
                    CAST(action_stats.completed_count AS FLOAT)
                    / NULLIF(action_stats.total_count, 0),
                    0.0
                ) AS utilization_score,
                'batch_derived' AS utilization_method,
                -- agent_match_score: NULL until expected-agent tracking
                -- is implemented; without a true expected agent the
                -- match score is meaningless.
                NULL AS agent_match_score,
                -- expected_agent: NULL because the true expected agent
                -- is not available in routing decisions data.
                NULL AS expected_agent,
                rd.selected_agent AS actual_agent,
                -- user_visible_latency: max action duration
                action_stats.max_duration_ms AS user_visible_latency_ms,
                rd.created_at,
                NOW() AS updated_at
            FROM agent_routing_decisions rd
            LEFT JOIN LATERAL (
                SELECT
                    COUNT(*) AS total_count,
                    COUNT(*) FILTER (WHERE aa.status = 'completed') AS completed_count,
                    MAX(aa.duration_ms) AS max_duration_ms
                FROM agent_actions aa
                WHERE aa.correlation_id = rd.correlation_id
            ) action_stats ON TRUE
            WHERE NOT EXISTS (
                SELECT 1 FROM injection_effectiveness ie
                WHERE ie.session_id = rd.correlation_id
            )
            ORDER BY rd.created_at DESC
            LIMIT $1
            ON CONFLICT (session_id) DO NOTHING
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await set_statement_timeout(conn, self._query_timeout * 1000)

                result: str = await conn.execute(sql, self._batch_size)

        # asyncpg execute returns "INSERT 0 N" string
        count = _parse_execute_count(result)

        logger.debug(
            "Computed injection_effectiveness rows",
            extra={
                "correlation_id": str(correlation_id),
                "rows_written": count,
            },
        )
        return count

    async def _compute_latency_breakdowns(self, correlation_id: UUID) -> int:
        """Derive latency_breakdowns from agent action durations.

        Each agent_action with a duration_ms value becomes a row in
        latency_breakdowns, using the action's correlation_id as session_id
        and the action id as prompt_id.

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows written.
        """
        sql = """
            INSERT INTO latency_breakdowns (
                session_id, prompt_id, cohort, cache_hit,
                routing_latency_ms, retrieval_latency_ms, injection_latency_ms,
                user_latency_ms, emitted_at, created_at
            )
            SELECT
                aa.correlation_id AS session_id,
                aa.id AS prompt_id,
                -- Derive cohort from routing confidence if available
                CASE
                    WHEN rd.confidence_score >= 0.8 THEN 'treatment'
                    WHEN rd.confidence_score IS NOT NULL THEN 'control'
                    ELSE NULL
                END AS cohort,
                FALSE AS cache_hit,
                -- Sub-component latencies are NULL because individual
                -- routing/retrieval/injection timing is not yet
                -- instrumented. Only total duration is available.
                NULL AS routing_latency_ms,
                NULL AS retrieval_latency_ms,
                NULL AS injection_latency_ms,
                COALESCE(aa.duration_ms, 0) AS user_latency_ms,
                aa.created_at AS emitted_at,
                NOW() AS created_at
            FROM agent_actions aa
            LEFT JOIN LATERAL (
                SELECT sub.confidence_score
                FROM agent_routing_decisions sub
                WHERE sub.correlation_id = aa.correlation_id
                ORDER BY sub.created_at DESC
                LIMIT 1
            ) rd ON TRUE
            WHERE aa.duration_ms IS NOT NULL
                AND aa.duration_ms > 0
                AND NOT EXISTS (
                    SELECT 1 FROM latency_breakdowns lb
                    WHERE lb.session_id = aa.correlation_id
                        AND lb.prompt_id = aa.id
                )
            ORDER BY aa.created_at DESC
            LIMIT $1
            ON CONFLICT (session_id, prompt_id) DO NOTHING
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await set_statement_timeout(conn, self._query_timeout * 1000)

                result: str = await conn.execute(sql, self._batch_size)

        count = _parse_execute_count(result)

        logger.debug(
            "Computed latency_breakdowns rows",
            extra={
                "correlation_id": str(correlation_id),
                "rows_written": count,
            },
        )
        return count

    async def _compute_pattern_hit_rates(self, correlation_id: UUID) -> int:
        """Derive pattern_hit_rates from agent selection patterns.

        Treats each unique per-agent selection as a "pattern" and
        computes hit rates based on how often that routing pattern led
        to successful actions.

        The pattern_id is derived deterministically from the selected_agent
        string using uuid5 with a fixed namespace.

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows written.
        """
        # Use a fixed namespace UUID for deterministic pattern_id generation
        # from agent names. This ensures the same agent always maps to the
        # same pattern_id across runs.
        sql = """
            INSERT INTO pattern_hit_rates (
                pattern_id, utilization_method, utilization_score,
                hit_count, miss_count, sample_count,
                created_at, updated_at
            )
            SELECT
                -- Deterministic UUID from agent name using MD5 hash cast
                uuid_generate_v5(
                    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'::uuid,
                    rd.selected_agent
                ) AS pattern_id,
                'batch_derived' AS utilization_method,
                -- utilization_score: average confidence for this agent
                AVG(rd.confidence_score) AS utilization_score,
                -- hit_count: routing decisions with high confidence
                COUNT(*) FILTER (WHERE rd.confidence_score >= 0.7) AS hit_count,
                -- miss_count: routing decisions with low confidence
                COUNT(*) FILTER (WHERE rd.confidence_score < 0.7) AS miss_count,
                -- sample_count: total decisions for this agent
                COUNT(*) AS sample_count,
                MIN(rd.created_at) AS created_at,
                NOW() AS updated_at
            FROM agent_routing_decisions rd
            GROUP BY rd.selected_agent
            LIMIT $1
            ON CONFLICT (pattern_id, utilization_method) DO UPDATE SET
                -- Counts are full snapshots (not accumulated), so the
                -- score must also be a snapshot to stay consistent.
                utilization_score = EXCLUDED.utilization_score,
                hit_count = EXCLUDED.hit_count,
                miss_count = EXCLUDED.miss_count,
                sample_count = EXCLUDED.sample_count,
                confidence = CASE
                    WHEN EXCLUDED.sample_count >= 20
                    THEN EXCLUDED.utilization_score
                    ELSE NULL
                END,
                updated_at = NOW()
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await set_statement_timeout(conn, self._query_timeout * 1000)

                # Check if uuid-ossp extension is available; if not, use a fallback.
                # Result is cached on the instance after the first query.
                if self._has_uuid_ossp is None:
                    row = await conn.fetchrow(
                        "SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp'"
                    )
                    self._has_uuid_ossp = row is not None

                if self._has_uuid_ossp:
                    result: str = await conn.execute(sql, self._batch_size)
                else:
                    # Fallback: use md5-based UUID generation without extension
                    sql_fallback = """
                        INSERT INTO pattern_hit_rates (
                            pattern_id, utilization_method, utilization_score,
                            hit_count, miss_count, sample_count,
                            created_at, updated_at
                        )
                        SELECT
                            md5(rd.selected_agent)::uuid AS pattern_id,
                            'batch_derived' AS utilization_method,
                            AVG(rd.confidence_score) AS utilization_score,
                            COUNT(*) FILTER (WHERE rd.confidence_score >= 0.7)
                                AS hit_count,
                            COUNT(*) FILTER (WHERE rd.confidence_score < 0.7)
                                AS miss_count,
                            COUNT(*) AS sample_count,
                            MIN(rd.created_at) AS created_at,
                            NOW() AS updated_at
                        FROM agent_routing_decisions rd
                        GROUP BY rd.selected_agent
                        LIMIT $1
                        ON CONFLICT (pattern_id, utilization_method) DO UPDATE SET
                            -- Counts are full snapshots (not accumulated), so the
                            -- score must also be a snapshot to stay consistent.
                            utilization_score = EXCLUDED.utilization_score,
                            hit_count = EXCLUDED.hit_count,
                            miss_count = EXCLUDED.miss_count,
                            sample_count = EXCLUDED.sample_count,
                            confidence = CASE
                                WHEN EXCLUDED.sample_count >= 20
                                THEN EXCLUDED.utilization_score
                                ELSE NULL
                            END,
                            updated_at = NOW()
                    """
                    result = await conn.execute(sql_fallback, self._batch_size)

        count = _parse_execute_count(result)

        logger.debug(
            "Computed pattern_hit_rates rows",
            extra={
                "correlation_id": str(correlation_id),
                "rows_written": count,
            },
        )
        return count


def _parse_execute_count(result: str) -> int:
    """Parse row count from asyncpg execute result string.

    asyncpg's execute() returns strings like "INSERT 0 42" or "UPDATE 42".

    Args:
        result: The result string from asyncpg execute().

    Returns:
        Number of affected rows, or 0 if parsing fails.
    """
    try:
        parts = result.split()
        return int(parts[-1])
    except (IndexError, ValueError):
        return 0


__all__ = [
    "BatchComputeEffectivenessMetrics",
    "BatchComputeResult",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_QUERY_TIMEOUT",
]
