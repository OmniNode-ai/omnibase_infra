# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that runs the 3-phase baselines batch computation as an EFFECT node.

Lifted from ServiceBatchComputeBaselines (OMN-3041). Follows the canonical
ONEX EFFECT handler pattern (mirrors HandlerRewardBinder, OMN-2927).

Key design decisions:
    D1: correlation_id is REQUIRED in the command (no default).
    D4: TREATMENT_CONFIDENCE_THRESHOLD imported from constants.py.
    D5: Emit snapshot only when sum(parse_execute_count across phases) > 0.
        No-op runs (all counts == 0) must NOT emit — avoids empty snapshots.
    D6: Uses publisher callable matching PublisherTopicScoped.publish signature.

Ticket: OMN-3044
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_command import (
    ModelBaselinesBatchComputeCommand,
)
from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_output import (
    ModelBaselinesBatchComputeOutput,
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

_TOPIC_BASELINES_COMPUTED = "onex.evt.omnibase-infra.baselines-computed.v1"
_EVENT_TYPE_BASELINES_COMPUTED = "baselines.computed"


@runtime_checkable
class ProtocolPublisher(Protocol):
    """Protocol matching PublisherTopicScoped.publish signature.

    Verified against omnibase_infra/runtime/publisher_topic_scoped.py:203.
    """

    async def __call__(
        self,
        event_type: str,
        payload: object,
        topic: str | None,
        correlation_id: object,
        **kwargs: object,
    ) -> bool: ...


class HandlerBaselinesBatchCompute:
    """EFFECT handler for 3-phase baselines batch computation.

    Lifted from ServiceBatchComputeBaselines. Runs the three computation
    phases (comparisons, trend, breakdown) and optionally emits a
    baselines-computed snapshot event to Kafka via injected publisher.

    The publisher callable is injected at construction time, enabling
    easy mocking in tests without touching Kafka infrastructure.

    Attributes:
        _pool: Injected asyncpg connection pool.
        _publisher: Optional async callable for publishing to Kafka.
        _batch_size: Row limit per phase.
        _query_timeout: Query timeout in seconds.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        publisher: Callable[..., Awaitable[bool]] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
    ) -> None:
        """Initialize the handler.

        Args:
            pool: asyncpg connection pool (lifecycle managed externally).
            publisher: Optional async callable matching ProtocolPublisher.
                When None, no snapshot event is emitted.
            batch_size: Row limit per SQL phase.
            query_timeout: Query timeout in seconds.
        """
        self._pool = pool
        self._publisher = publisher
        self._batch_size = batch_size
        self._query_timeout = query_timeout

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, command: ModelBaselinesBatchComputeCommand
    ) -> ModelBaselinesBatchComputeOutput:
        """Run the full baselines batch computation pipeline.

        Executes three computation phases sequentially:
            1. Daily comparisons (treatment vs control per day)
            2. Trend rows (per-cohort per-day time series)
            3. Breakdown rows (per-pattern treatment vs control)

        Partial snapshot policy (D5): emit only when
        sum(parse_execute_count across phases) > 0.
        No-op runs (all counts == 0) must NOT emit — avoids empty snapshots in omnidash.

        Args:
            command: Batch compute command with required correlation_id.

        Returns:
            ModelBaselinesBatchComputeOutput with result row counts
            and snapshot_emitted flag.
        """
        correlation_id = command.correlation_id
        started_at = datetime.now(UTC)
        errors: list[str] = []

        logger.info(
            "Starting baselines batch computation",
            extra={
                "correlation_id": str(correlation_id),
                "batch_size": self._batch_size,
            },
        )

        # Phase 1: baselines_comparisons (daily treatment vs control)
        comparisons_rows = 0
        try:
            comparisons_rows = await self._compute_comparisons(correlation_id)
        except Exception as e:
            safe_msg = sanitize_error_message(e)
            msg = f"Phase 1 (baselines_comparisons) failed: {safe_msg}"
            logger.exception(msg, extra={"correlation_id": str(correlation_id)})
            errors.append(msg)

        # Phase 2: baselines_trend (per-cohort per-day time series)
        trend_rows = 0
        try:
            trend_rows = await self._compute_trend(correlation_id)
        except Exception as e:
            safe_msg = sanitize_error_message(e)
            msg = f"Phase 2 (baselines_trend) failed: {safe_msg}"
            logger.exception(msg, extra={"correlation_id": str(correlation_id)})
            errors.append(msg)

        # Phase 3: baselines_breakdown (per-pattern treatment vs control)
        breakdown_rows = 0
        try:
            breakdown_rows = await self._compute_breakdown(correlation_id)
        except Exception as e:
            safe_msg = sanitize_error_message(e)
            msg = f"Phase 3 (baselines_breakdown) failed: {safe_msg}"
            logger.exception(msg, extra={"correlation_id": str(correlation_id)})
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
                "correlation_id": str(correlation_id),
                "comparisons_rows": comparisons_rows,
                "trend_rows": trend_rows,
                "breakdown_rows": breakdown_rows,
                "total_rows": result.total_rows,
                "has_errors": result.has_errors,
                "duration_seconds": (completed_at - started_at).total_seconds(),
            },
        )

        # D5: Emit snapshot only when sum(parse_execute_count across phases) > 0.
        # No-op runs (all counts == 0) must NOT emit — avoids empty snapshots in omnidash.
        snapshot_emitted = False
        if self._publisher is not None and result.total_rows > 0:
            try:
                await self._emit_snapshot(
                    result=result,
                    correlation_id=correlation_id,
                    computed_at=completed_at,
                    started_at=started_at,
                )
                snapshot_emitted = True
            except Exception as e:
                safe_msg = sanitize_error_message(e)
                msg = f"Snapshot emit failed: {safe_msg}"
                logger.warning(
                    "Failed to emit baselines-computed snapshot (non-fatal): %s",
                    safe_msg,
                    extra={"correlation_id": str(correlation_id)},
                )
                errors.append(msg)
                # Rebuild result with the additional error
                result = ModelBatchComputeBaselinesResult(
                    comparisons_rows=comparisons_rows,
                    trend_rows=trend_rows,
                    breakdown_rows=breakdown_rows,
                    errors=tuple(errors),
                    started_at=started_at,
                    completed_at=completed_at,
                )

        return ModelBaselinesBatchComputeOutput(
            result=result,
            snapshot_emitted=snapshot_emitted,
        )

    async def _publish(
        self,
        *,
        event_type: str,
        topic: str,
        payload: object,
        correlation_id: UUID,
    ) -> None:
        """Publish via injected publisher. Errors propagate to caller.

        Args:
            event_type: Event type string.
            topic: Kafka topic to publish to.
            payload: JSON-serializable payload dict.
            correlation_id: Correlation ID for tracing.
        """
        assert self._publisher is not None
        await self._publisher(
            event_type=event_type,
            payload=payload,
            topic=topic,
            correlation_id=correlation_id,
        )

    async def _emit_snapshot(
        self,
        result: ModelBatchComputeBaselinesResult,
        correlation_id: UUID,
        computed_at: datetime,
        started_at: datetime,
    ) -> None:
        """Read back computed rows and emit baselines-computed snapshot event.

        Args:
            result: Computation result with row counts.
            correlation_id: Correlation ID for tracing.
            computed_at: When the batch computation completed.
            started_at: When the batch computation started (used as window_start).
        """
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

        payload = snapshot.model_dump(mode="json")

        await self._publish(
            event_type=_EVENT_TYPE_BASELINES_COMPUTED,
            topic=_TOPIC_BASELINES_COMPUTED,
            payload=payload,
            correlation_id=correlation_id,
        )

        logger.info(
            "Emitted baselines-computed snapshot event",
            extra={
                "snapshot_id": str(snapshot_id),
                "correlation_id": str(correlation_id),
                "comparisons": len(comparisons),
                "trend": len(trend),
                "breakdown": len(breakdown),
                "topic": _TOPIC_BASELINES_COMPUTED,
            },
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

        Treatment group: confidence_score >= TREATMENT_CONFIDENCE_THRESHOLD (D4)
        Control group: confidence_score < TREATMENT_CONFIDENCE_THRESHOLD OR NULL

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Number of rows written.
        """
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

        Groups agent_routing_decisions by selected_agent and computes
        treatment/control split metrics. Uses md5(selected_agent)::uuid
        for stable pattern identity.

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


__all__: list[str] = ["HandlerBaselinesBatchCompute", "ProtocolPublisher"]
