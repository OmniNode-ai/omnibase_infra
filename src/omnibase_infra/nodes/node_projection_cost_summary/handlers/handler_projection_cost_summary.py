# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for repeatable cost summary snapshot emission."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from omniintelligence.models.events import ModelCostSummarySnapshot

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.services.cost_api.snapshot_cache import (
    TOPIC_COST_SUMMARY,
    store_latest_snapshot,
)

WINDOWS: tuple[str, ...] = ("24h", "7d", "30d")


class SnapshotPublisher(Protocol):
    async def publish(self, topic: str, payload: dict[str, object]) -> object:
        raise NotImplementedError


def _decimal(value: object) -> Decimal:
    if value is None:
        return Decimal("0.000000")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal | float):
        return int(value)
    return int(str(value))


def _row_get(row: Mapping[str, object] | None, key: str) -> object:
    if row is None:
        return None
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


class HandlerProjectionCostSummary:
    """Compute and publish cost summary snapshots from projection tables."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.PROJECTION_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def emit_snapshot(
        self,
        pool: object,
        publisher: SnapshotPublisher | None = None,
        *,
        window: str = "24h",
        snapshot_timestamp: datetime | None = None,
    ) -> ModelCostSummarySnapshot:
        """Compute one summary snapshot and optionally publish it."""
        timestamp = snapshot_timestamp or datetime.now(tz=UTC)
        async with pool.acquire() as conn:  # type: ignore[attr-defined]
            cost_row = await conn.fetchrow(
                """
                SELECT
                    COALESCE(SUM(total_cost_usd), 0)::numeric(14, 6) AS total_cost_usd,
                    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
                    COALESCE(SUM(call_count), 0)::bigint AS call_count
                FROM llm_cost_aggregates
                WHERE "window" = $1::cost_aggregation_window
                  AND aggregation_key LIKE 'session:%'
                """,
                window,
            )
            savings_row = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(savings_usd), 0)::numeric(14, 6) AS total_savings_usd
                FROM savings_estimates
                WHERE event_timestamp >= $1::timestamptz - (
                    CASE $2
                        WHEN '24h' THEN INTERVAL '24 hours'
                        WHEN '7d' THEN INTERVAL '7 days'
                        WHEN '30d' THEN INTERVAL '30 days'
                    END
                )
                  AND event_timestamp <= $1::timestamptz
                """,
                timestamp,
                window,
            )

        snapshot = ModelCostSummarySnapshot(
            window=window,  # type: ignore[arg-type]
            total_cost_usd=_decimal(_row_get(cost_row, "total_cost_usd")),
            total_savings_usd=_decimal(_row_get(savings_row, "total_savings_usd")),
            total_tokens=_int(_row_get(cost_row, "total_tokens")),
            session_count=_int(_row_get(cost_row, "call_count")),
            snapshot_timestamp=timestamp,
        )
        payload = snapshot.model_dump(mode="json")
        store_latest_snapshot(TOPIC_COST_SUMMARY, window, payload)
        if publisher is not None:
            await publisher.publish(TOPIC_COST_SUMMARY, payload)
        return snapshot


__all__ = ["HandlerProjectionCostSummary", "SnapshotPublisher", "WINDOWS"]
