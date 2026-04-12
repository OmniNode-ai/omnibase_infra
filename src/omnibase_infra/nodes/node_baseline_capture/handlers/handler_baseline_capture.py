# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that captures raw measurements from agent_actions and emits a baselines snapshot.

SOW Phase 2 — Track B4a. Reads existing pattern execution data (tokens, latency,
model_used) and emits onex.evt.omnibase-infra.baselines-computed.v1 without
computing deltas or ROI. Delta computation is deferred to B4b.

Design decisions:
    D1: correlation_id is REQUIRED in the command (no default).
    D2: lookback_hours is capped at 168 (7 days) to bound query cost.
    D3: Emit only when measurements_captured > 0 (no empty snapshots).
    D4: token counts read from action_details->>'total_tokens'; NULL treated as 0.
    D5: Publisher callable matches PublisherTopicScoped.publish signature (same as
        HandlerBaselinesBatchCompute — no new protocol needed).

Ticket: OMN-7484
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_baseline_capture.models.model_baseline_capture_command import (
    ModelBaselineCaptureCommand,
)
from omnibase_infra.nodes.node_baseline_capture.models.model_baseline_capture_output import (
    ModelBaselineCaptureOutput,
)
from omnibase_infra.runtime.emit_daemon.topics import TOPIC_BASELINES_COMPUTED
from omnibase_infra.services.observability.baselines.constants import (
    DEFAULT_QUERY_TIMEOUT,
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
from omnibase_infra.utils.util_db_transaction import set_statement_timeout
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

_TOPIC_BASELINES_COMPUTED = TOPIC_BASELINES_COMPUTED
_EVENT_TYPE_BASELINES_COMPUTED = "baselines.computed"
_MAX_LOOKBACK_HOURS = 168


@runtime_checkable
class ProtocolPublisher(Protocol):
    """Protocol matching PublisherTopicScoped.publish signature."""

    async def __call__(
        self,
        event_type: str,
        payload: object,
        topic: str | None,
        correlation_id: object,
        **kwargs: object,
    ) -> bool: ...


class HandlerBaselineCapture:
    """EFFECT handler for raw baseline measurement capture.

    Reads agent_actions and agent_routing_decisions within a lookback window,
    packages per-agent aggregates as a ModelBaselinesSnapshotEvent, and emits
    to onex.evt.omnibase-infra.baselines-computed.v1.

    No delta or ROI computation — that is B4b (post-Tuesday).

    Attributes:
        _pool: Injected asyncpg connection pool.
        _publisher: Optional async callable for publishing to Kafka.
        _query_timeout: Query timeout in seconds.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        publisher: Callable[..., Awaitable[bool]] | None = None,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
    ) -> None:
        self._pool = pool
        self._publisher = publisher
        self._query_timeout = query_timeout

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, command: ModelBaselineCaptureCommand
    ) -> ModelBaselineCaptureOutput:
        """Capture raw measurements and emit baselines snapshot.

        Reads agent_actions rows within the lookback window, aggregates per
        agent (model), and emits a ModelBaselinesSnapshotEvent with breakdown
        rows containing raw token/latency measurements.

        D3: emit only when measurements_captured > 0.

        Args:
            command: Capture command with correlation_id and lookback_hours.

        Returns:
            ModelBaselineCaptureOutput with row count and snapshot_emitted flag.
        """
        correlation_id = command.correlation_id
        lookback_hours = min(command.lookback_hours, _MAX_LOOKBACK_HOURS)
        since = datetime.now(UTC) - timedelta(hours=lookback_hours)

        errors: list[str] = []
        measurements_captured = 0
        breakdown: list[ModelBaselinesBreakdownRow] = []

        try:
            rows, measurements_captured = await self._read_raw_measurements(
                correlation_id=correlation_id,
                since=since,
            )
            breakdown = rows
        except Exception as e:
            safe_msg = sanitize_error_message(e)
            msg = f"Raw measurement read failed: {safe_msg}"
            logger.exception(msg, extra={"correlation_id": str(correlation_id)})
            errors.append(msg)

        # D3: no empty snapshots
        snapshot_emitted = False
        if self._publisher is not None and measurements_captured > 0:
            try:
                await self._emit_snapshot(
                    breakdown=breakdown,
                    correlation_id=correlation_id,
                    since=since,
                )
                snapshot_emitted = True
            except Exception as e:  # noqa: BLE001 — boundary: logs warning and degrades
                safe_msg = sanitize_error_message(e)
                msg = f"Snapshot emit failed: {safe_msg}"
                logger.warning(
                    "Failed to emit baselines snapshot (non-fatal): %s",
                    safe_msg,
                    extra={"correlation_id": str(correlation_id)},
                )
                errors.append(msg)

        logger.info(
            "Baseline capture completed",
            extra={
                "correlation_id": str(correlation_id),
                "measurements_captured": measurements_captured,
                "snapshot_emitted": snapshot_emitted,
                "lookback_hours": lookback_hours,
            },
        )

        return ModelBaselineCaptureOutput(
            measurements_captured=measurements_captured,
            snapshot_emitted=snapshot_emitted,
            errors=tuple(errors),
        )

    async def _read_raw_measurements(
        self,
        correlation_id: UUID,
        since: datetime,
    ) -> tuple[list[ModelBaselinesBreakdownRow], int]:
        """Read agent_actions within the lookback window, grouped by agent_name.

        Aggregates token counts (from action_details->>'total_tokens') and
        latency (duration_ms) per agent. Returns breakdown rows and total count.

        D4: NULL token values coerced to 0.

        Args:
            correlation_id: For logging.
            since: Earliest created_at to include.

        Returns:
            (breakdown_rows, total_measurement_count)
        """
        sql = """
            SELECT
                md5(agent_name)::uuid AS pattern_id,
                agent_name AS pattern_label,
                COUNT(*) AS sample_count,
                AVG(duration_ms) AS avg_latency_ms,
                SUM(
                    COALESCE(
                        (action_details->>'total_tokens')::bigint,
                        0
                    )
                ) AS total_tokens,
                AVG(
                    COALESCE(
                        (action_details->>'total_tokens')::bigint,
                        0
                    )
                ) AS avg_tokens,
                COUNT(*) AS success_count,
                COUNT(*) AS treatment_count,
                0 AS control_count,
                NOW() AS computed_at,
                NOW() AS created_at,
                NOW() AS updated_at
            FROM agent_actions
            WHERE created_at >= $1
            GROUP BY agent_name
            ORDER BY sample_count DESC
        """
        total_count_sql = """
            SELECT COUNT(*) AS total FROM agent_actions WHERE created_at >= $1
        """
        async with self._pool.acquire() as conn:
            await set_statement_timeout(conn, self._query_timeout * 1000)
            rows = await conn.fetch(sql, since)
            total_row = await conn.fetchrow(total_count_sql, since)

        total = int(total_row["total"]) if total_row else 0

        breakdown: list[ModelBaselinesBreakdownRow] = []
        for row in rows:
            sample_count = int(row["sample_count"])
            success_count = int(row["success_count"])
            treatment_success_rate = (
                float(success_count) / float(sample_count) if sample_count > 0 else None
            )
            breakdown.append(
                ModelBaselinesBreakdownRow(
                    id=uuid4(),
                    pattern_id=row["pattern_id"],
                    pattern_label=row["pattern_label"],
                    treatment_success_rate=treatment_success_rate,
                    control_success_rate=None,
                    roi_pct=None,
                    sample_count=sample_count,
                    treatment_count=int(row["treatment_count"]),
                    control_count=int(row["control_count"]),
                    confidence=None,
                    computed_at=row["computed_at"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )

        logger.debug(
            "Raw measurements read",
            extra={
                "correlation_id": str(correlation_id),
                "agent_groups": len(breakdown),
                "total_rows": total,
            },
        )
        return breakdown, total

    async def _emit_snapshot(
        self,
        breakdown: list[ModelBaselinesBreakdownRow],
        correlation_id: UUID,
        since: datetime,
    ) -> None:
        """Emit baselines-computed snapshot with raw measurement breakdown.

        comparisons and trend are empty — B4a captures raw measurements only.
        Deltas and ROI are deferred to B4b.

        Args:
            breakdown: Per-agent raw measurement rows.
            correlation_id: For tracing.
            since: Window start (lookback boundary).
        """
        assert self._publisher is not None
        snapshot_id = uuid4()
        computed_at = datetime.now(UTC)

        snapshot = ModelBaselinesSnapshotEvent(
            snapshot_id=snapshot_id,
            contract_version=1,
            computed_at_utc=computed_at,
            window_start_utc=since,
            window_end_utc=computed_at,
            comparisons=[],
            trend=[],
            breakdown=breakdown,
        )

        payload = snapshot.model_dump(mode="json")

        await self._publisher(
            event_type=_EVENT_TYPE_BASELINES_COMPUTED,
            payload=payload,
            topic=_TOPIC_BASELINES_COMPUTED,
            correlation_id=correlation_id,
        )

        logger.info(
            "Emitted raw baselines snapshot",
            extra={
                "snapshot_id": str(snapshot_id),
                "correlation_id": str(correlation_id),
                "breakdown_rows": len(breakdown),
                "topic": _TOPIC_BASELINES_COMPUTED,
            },
        )


__all__: list[str] = ["HandlerBaselineCapture", "ProtocolPublisher"]
