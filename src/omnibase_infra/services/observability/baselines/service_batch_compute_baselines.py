# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Batch computation for baselines treatment/control comparisons.

Derives A/B comparison data from the existing ``agent_routing_decisions``
and ``agent_actions`` tables, populating the three baselines tables:
``baselines_comparisons``, ``baselines_trend``, and ``baselines_breakdown``.

This service bridges the gap described in OMN-2305: the Baselines & ROI
page (``/baselines``) falls back to mock data when the baselines tables
are empty. This service seeds those tables with real comparison data
derived from existing observability data.

Treatment vs Control Definition:
    - **Treatment**: ``agent_routing_decisions`` rows with
      ``confidence_score >= 0.8`` -- high-confidence selections indicating
      active pattern injection context.
    - **Control**: ``agent_routing_decisions`` rows with
      ``confidence_score < 0.8`` or ``NULL`` -- low-confidence or no
      injection context.

This definition mirrors the ``cohort`` classification already used in
the ``injection_effectiveness`` table, ensuring consistent A/B labeling
across the observability stack.

ROI Formula:
    ``roi_pct = (treatment_success_rate - control_success_rate)
               / control_success_rate * 100``

    NULL when ``control_success_rate`` is zero or NULL (SQL handles
    via ``NULLIF``).

Design Decisions:
    - Read from agent_routing_decisions + agent_actions (already populated)
    - Write to baselines_comparisons, baselines_trend, baselines_breakdown
    - Idempotent: ON CONFLICT DO UPDATE for all three tables
    - Pool injection: asyncpg.Pool injected, lifecycle managed externally
    - Phase isolation: each phase failure is caught and logged independently

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons

Example:
    >>> import asyncpg
    >>> from omnibase_infra.services.observability.baselines.service_batch_compute_baselines import (
    ...     ServiceBatchComputeBaselines,
    ... )
    >>>
    >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
    >>> batch = ServiceBatchComputeBaselines(pool)
    >>> result = await batch.compute_and_persist()
    >>> print(f"Wrote {result.comparisons_rows} comparison rows")
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.emit_daemon.event_registry import EventRegistry
from omnibase_infra.runtime.emit_daemon.topics import (
    BASELINES_COMPUTED_REGISTRATION,
    TOPIC_BASELINES_COMPUTED,
)
from omnibase_infra.services.observability.baselines.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_QUERY_TIMEOUT,
    TREATMENT_CONFIDENCE_THRESHOLD,
    parse_execute_count,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_breakdown_row import (
    ModelBaselinesBreakdownRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_comparison_row import (
    ModelBaselinesComparisonRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_snapshot_event import (
    ModelBaselinesSnapshotEvent,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_trend_row import (
    ModelBaselinesTrendRow,
)
from omnibase_infra.services.observability.baselines.models.model_batch_compute_baselines_result import (
    ModelBatchComputeBaselinesResult,
)
from omnibase_infra.utils.util_db_transaction import set_statement_timeout
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

_EMIT_REGISTRY = EventRegistry()
_EMIT_REGISTRY.register(BASELINES_COMPUTED_REGISTRATION)


class ServiceBatchComputeBaselines:
    """Batch computation engine for baselines treatment/control comparisons.

    .. deprecated::
        Use ``HandlerBaselinesBatchCompute`` instead. This service is
        superseded by the canonical ONEX EFFECT node implementation
        (OMN-3039). It will be removed in a future release.

        Migration: replace ``ServiceBatchComputeBaselines(pool, event_bus=bus)``
        with ``HandlerBaselinesBatchCompute(pool, publisher=pub.publish)``.

    Reads existing data from agent_routing_decisions and agent_actions
    and derives treatment vs control comparison data for the three
    baselines tables.

    The computation is idempotent: running it multiple times produces
    the same result due to ON CONFLICT DO UPDATE handling.

    Attributes:
        _pool: Injected asyncpg connection pool.
        _batch_size: Limit for per-phase SQL queries.
        _query_timeout: Query timeout in seconds.

    Example:
        >>> pool = await asyncpg.create_pool(dsn="postgresql://...")
        >>> batch = ServiceBatchComputeBaselines(pool, batch_size=200)
        >>> result = await batch.compute_and_persist()
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
        event_bus: ProtocolEventBusLike | None = None,
    ) -> None:
        """Initialize batch computation engine.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            batch_size: Row limit per phase.
            query_timeout: Query timeout in seconds.
            event_bus: Optional event bus for emitting the baselines-computed
                snapshot event. When None, no event is emitted (e.g. during
                migrations or offline batch runs). Lifecycle managed externally.
        """
        import warnings

        warnings.warn(
            "ServiceBatchComputeBaselines is deprecated. "
            "Use HandlerBaselinesBatchCompute instead (OMN-3039).",
            DeprecationWarning,
            stacklevel=2,
        )
        self._pool = pool
        self._batch_size = batch_size
        self._query_timeout = query_timeout
        self._event_bus = event_bus

    async def compute_and_persist(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelBatchComputeBaselinesResult:
        """Run the full baselines batch computation pipeline.

        Executes three computation phases sequentially:
            1. Daily comparisons (treatment vs control per day)
            2. Trend rows (per-cohort per-day time series)
            3. Breakdown rows (per-pattern treatment vs control)

        All writes are idempotent (ON CONFLICT DO UPDATE).
        Individual phase failures are caught and recorded in the result's
        ``errors`` tuple rather than raised, so subsequent phases still run.

        Args:
            correlation_id: Optional correlation ID for tracing. A new
                UUID is generated if not provided.

        Returns:
            ModelBatchComputeBaselinesResult with per-table row counts,
            any phase error messages, and timestamps.
        """
        effective_correlation_id = correlation_id or uuid4()
        started_at = datetime.now(UTC)
        errors: list[str] = []

        logger.info(
            "Starting baselines batch computation",
            extra={
                "correlation_id": str(effective_correlation_id),
                "batch_size": self._batch_size,
            },
        )

        # Phase 1: baselines_comparisons (daily treatment vs control)
        comparisons_rows = 0
        try:
            comparisons_rows = await self._compute_comparisons(effective_correlation_id)
        except Exception as e:
            safe_msg = sanitize_error_message(e)
            msg = f"Phase 1 (baselines_comparisons) failed: {safe_msg}"
            logger.exception(
                msg, extra={"correlation_id": str(effective_correlation_id)}
            )
            errors.append(msg)

        # Phase 2: baselines_trend (per-cohort per-day time series)
        trend_rows = 0
        try:
            trend_rows = await self._compute_trend(effective_correlation_id)
        except Exception as e:
            safe_msg = sanitize_error_message(e)
            msg = f"Phase 2 (baselines_trend) failed: {safe_msg}"
            logger.exception(
                msg, extra={"correlation_id": str(effective_correlation_id)}
            )
            errors.append(msg)

        # Phase 3: baselines_breakdown (per-pattern treatment vs control)
        breakdown_rows = 0
        try:
            breakdown_rows = await self._compute_breakdown(effective_correlation_id)
        except Exception as e:
            safe_msg = sanitize_error_message(e)
            msg = f"Phase 3 (baselines_breakdown) failed: {safe_msg}"
            logger.exception(
                msg, extra={"correlation_id": str(effective_correlation_id)}
            )
            errors.append(msg)

        completed_at = datetime.now(UTC)

        result = ModelBatchComputeBaselinesResult(
            comparisons_rows=comparisons_rows,
            trend_rows=trend_rows,
            breakdown_rows=breakdown_rows,
            errors=tuple(errors),
            started_at=started_at,
            completed_at=completed_at,
        )

        logger.info(
            "Baselines batch computation completed",
            extra={
                "correlation_id": str(effective_correlation_id),
                "comparisons_rows": comparisons_rows,
                "trend_rows": trend_rows,
                "breakdown_rows": breakdown_rows,
                "total_rows": result.total_rows,
                "has_errors": result.has_errors,
                "duration_seconds": (completed_at - started_at).total_seconds(),
            },
        )

        if self._event_bus is not None:
            total_rows = result.total_rows
            if total_rows == 0:
                logger.warning(
                    "Skipping baselines snapshot emit: no rows written in any phase",
                    extra={
                        "correlation_id": str(effective_correlation_id),
                        "has_errors": result.has_errors,
                    },
                )
            else:
                await self._emit_snapshot(
                    correlation_id=effective_correlation_id,
                    computed_at=completed_at,
                    started_at=started_at,
                )

        return result

    async def _emit_snapshot(
        self,
        correlation_id: UUID,
        computed_at: datetime,
        started_at: datetime,
    ) -> None:
        """Read back the computed rows and emit a baselines-computed snapshot event.

        Best-effort: logs a warning on failure but does not raise, because the
        DB write already committed successfully before this is called.

        Args:
            correlation_id: Correlation ID for tracing.
            computed_at: When the batch computation completed.
            started_at: When the batch computation started (used as window_start).
        """
        if self._event_bus is None:
            return

        try:
            snapshot_id = uuid4()

            comparisons = await self._read_comparisons()
            trend = await self._read_trend()
            breakdown = await self._read_breakdown()

            window_start = started_at if comparisons or trend else None
            window_end = computed_at if comparisons or trend else None

            snapshot = ModelBaselinesSnapshotEvent(
                snapshot_id=snapshot_id,
                contract_version=1,
                computed_at_utc=computed_at,
                window_start_utc=window_start,
                window_end_utc=window_end,
                comparisons=comparisons,
                trend=trend,
                breakdown=breakdown,
            )

            payload = _EMIT_REGISTRY.inject_metadata(
                event_type=BASELINES_COMPUTED_REGISTRATION.event_type,
                payload=snapshot.model_dump(mode="json"),
                correlation_id=str(correlation_id),
            )

            envelope: ModelEventEnvelope[dict[str, object]] = ModelEventEnvelope(
                payload=payload,
                correlation_id=correlation_id,
                source_tool="ServiceBatchComputeBaselines",
            )

            await self._event_bus.publish_envelope(envelope, TOPIC_BASELINES_COMPUTED)

            logger.info(
                "Emitted baselines-computed snapshot event",
                extra={
                    "snapshot_id": str(snapshot_id),
                    "correlation_id": str(correlation_id),
                    "comparisons": len(comparisons),
                    "trend": len(trend),
                    "breakdown": len(breakdown),
                    "topic": TOPIC_BASELINES_COMPUTED,
                },
            )

        except Exception as e:
            safe_msg = sanitize_error_message(e)
            logger.warning(
                "Failed to emit baselines-computed snapshot event (non-fatal): %s",
                safe_msg,
                extra={"correlation_id": str(correlation_id)},
            )

    async def _read_comparisons(self) -> list[ModelBaselinesComparisonRow]:
        """Read back rows from baselines_comparisons, ordered by date descending."""
        sql = """
            SELECT
                id, comparison_date, period_label,
                treatment_sessions, treatment_success_rate,
                treatment_avg_latency_ms, treatment_avg_cost_tokens,
                treatment_total_tokens,
                control_sessions, control_success_rate,
                control_avg_latency_ms, control_avg_cost_tokens,
                control_total_tokens,
                roi_pct, latency_improvement_pct, cost_improvement_pct,
                sample_size, computed_at, created_at, updated_at
            FROM baselines_comparisons
            ORDER BY comparison_date DESC
            LIMIT $1
        """
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            rows = await conn.fetch(sql, self._batch_size)
        return [ModelBaselinesComparisonRow(**dict(row)) for row in rows]

    async def _read_trend(self) -> list[ModelBaselinesTrendRow]:
        """Read back rows from baselines_trend, ordered by date descending."""
        sql = """
            SELECT
                id, trend_date, cohort,
                session_count, success_rate,
                avg_latency_ms, avg_cost_tokens,
                roi_pct, computed_at, created_at
            FROM baselines_trend
            ORDER BY trend_date DESC, cohort
            LIMIT $1
        """
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            rows = await conn.fetch(sql, self._batch_size)
        return [ModelBaselinesTrendRow(**dict(row)) for row in rows]

    async def _read_breakdown(self) -> list[ModelBaselinesBreakdownRow]:
        """Read back rows from baselines_breakdown, ordered by roi_pct descending."""
        sql = """
            SELECT
                id, pattern_id, pattern_label,
                treatment_success_rate, control_success_rate,
                roi_pct, sample_count, treatment_count, control_count,
                confidence, computed_at, created_at, updated_at
            FROM baselines_breakdown
            ORDER BY roi_pct DESC NULLS LAST
            LIMIT $1
        """
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            rows = await conn.fetch(sql, self._batch_size)
        return [ModelBaselinesBreakdownRow(**dict(row)) for row in rows]

    async def _compute_comparisons(self, correlation_id: UUID) -> int:
        """Derive daily treatment vs control comparison rows.

        For each distinct date in agent_routing_decisions, computes
        treatment and control group metrics and writes one row per day
        to baselines_comparisons.

        Treatment group: confidence_score >= 0.8
        Control group: confidence_score < 0.8 OR NULL

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows written.
        """
        # This query:
        # 1. Groups routing decisions by DATE(created_at)
        # 2. Splits each day into treatment (confidence >= 0.8) and control groups
        # 3. Joins agent_actions via LATERAL to compute per-session success rates
        # 4. Aggregates to daily treatment/control metrics
        # 5. Derives ROI as (treatment_success - control_success) / control_success
        # 6. Upserts with ON CONFLICT DO UPDATE to ensure idempotency
        sql = """
            INSERT INTO baselines_comparisons (
                comparison_date, period_label,
                treatment_sessions, treatment_success_rate,
                treatment_avg_latency_ms, treatment_avg_cost_tokens,
                treatment_total_tokens,
                control_sessions, control_success_rate,
                control_avg_latency_ms, control_avg_cost_tokens,
                control_total_tokens,
                roi_pct, latency_improvement_pct, cost_improvement_pct,
                sample_size,
                computed_at, created_at, updated_at
            )
            WITH daily_routing AS (
                SELECT
                    DATE(rd.created_at) AS comparison_date,
                    rd.correlation_id,
                    CASE
                        WHEN rd.confidence_score >= $2 THEN 'treatment'
                        ELSE 'control'
                    END AS cohort,
                    rd.confidence_score,
                    action_stats.success_rate AS session_success_rate,
                    action_stats.avg_duration_ms,
                    action_stats.total_tokens
                FROM agent_routing_decisions rd
                LEFT JOIN LATERAL (
                    SELECT
                        COALESCE(
                            CAST(COUNT(*) FILTER (WHERE aa.status = 'completed') AS FLOAT)
                            / NULLIF(COUNT(*), 0),
                            0.0
                        ) AS success_rate,
                        AVG(aa.duration_ms) AS avg_duration_ms,
                        SUM(aa.total_tokens) AS total_tokens
                    FROM agent_actions aa
                    WHERE aa.correlation_id = rd.correlation_id
                ) action_stats ON TRUE
                WHERE rd.correlation_id IS NOT NULL
                    AND rd.created_at >= NOW() - INTERVAL '90 days'
            ),
            daily_agg AS (
                SELECT
                    comparison_date,
                    -- Treatment group
                    COUNT(*) FILTER (WHERE cohort = 'treatment')
                        AS treatment_sessions,
                    AVG(session_success_rate) FILTER (WHERE cohort = 'treatment')
                        AS treatment_success_rate,
                    AVG(avg_duration_ms) FILTER (WHERE cohort = 'treatment')
                        AS treatment_avg_latency_ms,
                    AVG(total_tokens) FILTER (WHERE cohort = 'treatment')
                        AS treatment_avg_cost_tokens,
                    COALESCE(
                        SUM(total_tokens) FILTER (WHERE cohort = 'treatment'), 0
                    ) AS treatment_total_tokens,
                    -- Control group
                    COUNT(*) FILTER (WHERE cohort = 'control')
                        AS control_sessions,
                    AVG(session_success_rate) FILTER (WHERE cohort = 'control')
                        AS control_success_rate,
                    AVG(avg_duration_ms) FILTER (WHERE cohort = 'control')
                        AS control_avg_latency_ms,
                    AVG(total_tokens) FILTER (WHERE cohort = 'control')
                        AS control_avg_cost_tokens,
                    COALESCE(
                        SUM(total_tokens) FILTER (WHERE cohort = 'control'), 0
                    ) AS control_total_tokens
                FROM daily_routing
                GROUP BY comparison_date
                ORDER BY comparison_date DESC
                LIMIT $1
            )
            SELECT
                comparison_date,
                comparison_date::TEXT AS period_label,
                treatment_sessions,
                treatment_success_rate,
                treatment_avg_latency_ms,
                treatment_avg_cost_tokens,
                treatment_total_tokens,
                control_sessions,
                control_success_rate,
                control_avg_latency_ms,
                control_avg_cost_tokens,
                control_total_tokens,
                -- ROI: (treatment - control) / control * 100
                CASE
                    WHEN control_success_rate IS NOT NULL
                        AND control_success_rate > 0
                    THEN (treatment_success_rate - control_success_rate)
                         / control_success_rate * 100.0
                    ELSE NULL
                END AS roi_pct,
                -- Latency improvement: (control - treatment) / control * 100
                CASE
                    WHEN control_avg_latency_ms IS NOT NULL
                        AND control_avg_latency_ms > 0
                    THEN (control_avg_latency_ms - treatment_avg_latency_ms)
                         / control_avg_latency_ms * 100.0
                    ELSE NULL
                END AS latency_improvement_pct,
                -- Cost improvement: (control - treatment) / control * 100
                CASE
                    WHEN control_avg_cost_tokens IS NOT NULL
                        AND control_avg_cost_tokens > 0
                    THEN (control_avg_cost_tokens - treatment_avg_cost_tokens)
                         / control_avg_cost_tokens * 100.0
                    ELSE NULL
                END AS cost_improvement_pct,
                treatment_sessions + control_sessions AS sample_size,
                NOW() AS computed_at,
                NOW() AS created_at,
                NOW() AS updated_at
            FROM daily_agg
            ON CONFLICT (comparison_date) DO UPDATE SET
                period_label = EXCLUDED.period_label,
                treatment_sessions = EXCLUDED.treatment_sessions,
                treatment_success_rate = EXCLUDED.treatment_success_rate,
                treatment_avg_latency_ms = EXCLUDED.treatment_avg_latency_ms,
                treatment_avg_cost_tokens = EXCLUDED.treatment_avg_cost_tokens,
                treatment_total_tokens = EXCLUDED.treatment_total_tokens,
                control_sessions = EXCLUDED.control_sessions,
                control_success_rate = EXCLUDED.control_success_rate,
                control_avg_latency_ms = EXCLUDED.control_avg_latency_ms,
                control_avg_cost_tokens = EXCLUDED.control_avg_cost_tokens,
                control_total_tokens = EXCLUDED.control_total_tokens,
                roi_pct = EXCLUDED.roi_pct,
                latency_improvement_pct = EXCLUDED.latency_improvement_pct,
                cost_improvement_pct = EXCLUDED.cost_improvement_pct,
                sample_size = EXCLUDED.sample_size,
                computed_at = EXCLUDED.computed_at,
                updated_at = EXCLUDED.updated_at
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await set_statement_timeout(conn, self._query_timeout * 1000)
                result: str = await conn.execute(
                    sql, self._batch_size, TREATMENT_CONFIDENCE_THRESHOLD
                )

        count = parse_execute_count(result)
        logger.debug(
            "Computed baselines_comparisons rows",
            extra={
                "correlation_id": str(correlation_id),
                "rows_written": count,
            },
        )
        return count

    async def _compute_trend(self, correlation_id: UUID) -> int:
        """Derive per-cohort per-day trend rows.

        For each (cohort, date) pair in agent_routing_decisions, writes
        one row to baselines_trend containing that cohort's daily metrics.

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows written.
        """
        sql = """
            INSERT INTO baselines_trend (
                trend_date, cohort,
                session_count, success_rate,
                avg_latency_ms, avg_cost_tokens,
                roi_pct,
                computed_at, created_at
            )
            WITH daily_cohort AS (
                SELECT
                    DATE(rd.created_at) AS trend_date,
                    CASE
                        WHEN rd.confidence_score >= $2 THEN 'treatment'
                        ELSE 'control'
                    END AS cohort,
                    rd.correlation_id,
                    action_stats.success_rate,
                    action_stats.avg_duration_ms,
                    action_stats.total_tokens
                FROM agent_routing_decisions rd
                LEFT JOIN LATERAL (
                    SELECT
                        COALESCE(
                            CAST(COUNT(*) FILTER (WHERE aa.status = 'completed') AS FLOAT)
                            / NULLIF(COUNT(*), 0),
                            0.0
                        ) AS success_rate,
                        AVG(aa.duration_ms) AS avg_duration_ms,
                        SUM(aa.total_tokens) AS total_tokens
                    FROM agent_actions aa
                    WHERE aa.correlation_id = rd.correlation_id
                ) action_stats ON TRUE
                WHERE rd.correlation_id IS NOT NULL
                    AND rd.created_at >= NOW() - INTERVAL '90 days'
            ),
            cohort_agg AS (
                SELECT
                    trend_date,
                    cohort,
                    COUNT(*) AS session_count,
                    AVG(success_rate) AS success_rate,
                    AVG(avg_duration_ms) AS avg_latency_ms,
                    AVG(total_tokens) AS avg_cost_tokens
                FROM daily_cohort
                GROUP BY trend_date, cohort
            )
            SELECT
                trend_date,
                cohort,
                session_count,
                success_rate,
                avg_latency_ms,
                avg_cost_tokens,
                -- ROI relative to control for same day
                CASE
                    WHEN cohort = 'treatment' THEN (
                        SELECT
                            CASE
                                WHEN ctrl.success_rate > 0
                                THEN (ca.success_rate - ctrl.success_rate)
                                     / ctrl.success_rate * 100.0
                                ELSE NULL
                            END
                        FROM cohort_agg ctrl
                        WHERE ctrl.trend_date = ca.trend_date
                            AND ctrl.cohort = 'control'
                        LIMIT 1
                    )
                    ELSE NULL
                END AS roi_pct,
                NOW() AS computed_at,
                NOW() AS created_at
            FROM cohort_agg ca
            ORDER BY trend_date DESC, cohort
            LIMIT $1
            ON CONFLICT (trend_date, cohort) DO UPDATE SET
                session_count = EXCLUDED.session_count,
                success_rate = EXCLUDED.success_rate,
                avg_latency_ms = EXCLUDED.avg_latency_ms,
                avg_cost_tokens = EXCLUDED.avg_cost_tokens,
                roi_pct = EXCLUDED.roi_pct,
                computed_at = EXCLUDED.computed_at
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await set_statement_timeout(conn, self._query_timeout * 1000)
                result: str = await conn.execute(
                    sql, self._batch_size, TREATMENT_CONFIDENCE_THRESHOLD
                )

        count = parse_execute_count(result)
        logger.debug(
            "Computed baselines_trend rows",
            extra={
                "correlation_id": str(correlation_id),
                "rows_written": count,
            },
        )
        return count

    async def _compute_breakdown(self, correlation_id: UUID) -> int:
        """Derive per-pattern treatment vs control breakdown rows.

        Groups agent_routing_decisions by selected_agent (treated as a
        pattern proxy) and computes treatment/control split metrics.
        Uses md5(selected_agent)::uuid for stable pattern identity.

        Note:
            **Hard cap, not true batching**: Groups by selected_agent
            before applying LIMIT. If distinct agent count exceeds
            batch_size, agents beyond the cap are silently skipped.
            A warning is logged when the result count equals batch_size.

            **Alphabetical truncation bias**: The inner CTE uses
            ``ORDER BY selected_agent`` before ``LIMIT $1``. This means
            the hard cap excludes agents whose names sort alphabetically
            last, not agents with the fewest sessions. High-traffic agents
            may be silently excluded if their names sort late.

            **Time window**: Only agent_routing_decisions rows from the
            past 90 days are included, filtered inside the agent_sessions
            CTE via ``rd.created_at >= NOW() - INTERVAL '90 days'``.

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows written.
        """
        sql = """
            INSERT INTO baselines_breakdown (
                pattern_id, pattern_label,
                treatment_success_rate, control_success_rate,
                roi_pct, sample_count, treatment_count, control_count,
                confidence,
                computed_at, created_at, updated_at
            )
            WITH agent_sessions AS (
                SELECT
                    rd.selected_agent,
                    CASE
                        WHEN rd.confidence_score >= $2 THEN 'treatment'
                        ELSE 'control'
                    END AS cohort,
                    rd.correlation_id,
                    action_stats.success_rate
                FROM agent_routing_decisions rd
                LEFT JOIN LATERAL (
                    SELECT
                        COALESCE(
                            CAST(COUNT(*) FILTER (WHERE aa.status = 'completed') AS FLOAT)
                            / NULLIF(COUNT(*), 0),
                            0.0
                        ) AS success_rate
                    FROM agent_actions aa
                    WHERE aa.correlation_id = rd.correlation_id
                ) action_stats ON TRUE
                WHERE rd.selected_agent IS NOT NULL
                    AND rd.correlation_id IS NOT NULL
                    -- Time window: restrict to past 90 days, consistent with comparisons and trend phases.
                    AND rd.created_at >= NOW() - INTERVAL '90 days'
            ),
            agent_agg AS (
                SELECT
                    selected_agent,
                    COUNT(*) AS sample_count,
                    COUNT(*) FILTER (WHERE cohort = 'treatment') AS treatment_count,
                    COUNT(*) FILTER (WHERE cohort = 'control') AS control_count,
                    AVG(success_rate) FILTER (WHERE cohort = 'treatment')
                        AS treatment_success_rate,
                    AVG(success_rate) FILTER (WHERE cohort = 'control')
                        AS control_success_rate
                FROM agent_sessions
                GROUP BY selected_agent
                ORDER BY selected_agent
                LIMIT $1
            )
            SELECT
                md5(selected_agent)::uuid AS pattern_id,
                selected_agent AS pattern_label,
                treatment_success_rate,
                control_success_rate,
                -- ROI: (treatment - control) / control * 100
                CASE
                    WHEN control_success_rate IS NOT NULL
                        AND control_success_rate > 0
                    THEN (treatment_success_rate - control_success_rate)
                         / control_success_rate * 100.0
                    ELSE NULL
                END AS roi_pct,
                sample_count,
                treatment_count,
                control_count,
                -- Confidence proxy: treatment_success_rate when sample_count >= 20; NULL below threshold.
                -- Not a statistical confidence interval.
                CASE
                    WHEN sample_count >= 20
                        AND treatment_success_rate IS NOT NULL
                    THEN treatment_success_rate
                    ELSE NULL
                END AS confidence,
                NOW() AS computed_at,
                NOW() AS created_at,
                NOW() AS updated_at
            FROM agent_agg
            ON CONFLICT (pattern_id) DO UPDATE SET
                pattern_label = EXCLUDED.pattern_label,
                treatment_success_rate = EXCLUDED.treatment_success_rate,
                control_success_rate = EXCLUDED.control_success_rate,
                roi_pct = EXCLUDED.roi_pct,
                sample_count = EXCLUDED.sample_count,
                treatment_count = EXCLUDED.treatment_count,
                control_count = EXCLUDED.control_count,
                confidence = EXCLUDED.confidence,
                computed_at = EXCLUDED.computed_at,
                updated_at = EXCLUDED.updated_at
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await set_statement_timeout(conn, self._query_timeout * 1000)
                result: str = await conn.execute(
                    sql, self._batch_size, TREATMENT_CONFIDENCE_THRESHOLD
                )

        count = parse_execute_count(result)

        if count == self._batch_size:
            logger.warning(
                "baselines_breakdown phase returned exactly batch_size rows; "
                "some agents may have been truncated. "
                "Breakdown truncated to %d of at least %d agents "
                "(true distinct agent count may be higher). "
                "Increase batch_size if more agents should be included.",
                self._batch_size,
                self._batch_size + 1,
                extra={"correlation_id": str(correlation_id)},
            )

        logger.debug(
            "Computed baselines_breakdown rows",
            extra={
                "correlation_id": str(correlation_id),
                "rows_written": count,
            },
        )
        return count


__all__: list[str] = [
    "ServiceBatchComputeBaselines",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_QUERY_TIMEOUT",
    "TREATMENT_CONFIDENCE_THRESHOLD",
    "parse_execute_count",
]
