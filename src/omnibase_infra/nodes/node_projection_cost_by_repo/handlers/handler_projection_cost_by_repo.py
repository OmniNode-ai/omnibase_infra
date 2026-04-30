# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for repeatable cost-by-repo snapshot emission."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from omniintelligence.models.events import (
    ModelCostByRepoSnapshot,
    ModelCostByRepoSnapshotRow,
)

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.services.cost_api.snapshot_cache import (
    TOPIC_COST_BY_REPO,
    store_latest_snapshot,
)


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


def _row_get(row: Mapping[str, object], key: str) -> object:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


class HandlerProjectionCostByRepo:
    """Compute and publish repository cost snapshots from raw projection data."""

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
    ) -> ModelCostByRepoSnapshot:
        """Compute one by-repo snapshot and optionally publish it."""
        timestamp = snapshot_timestamp or datetime.now(tz=UTC)
        async with pool.acquire() as conn:  # type: ignore[attr-defined]
            rows = await conn.fetch(
                """
                SELECT
                    COALESCE(repo_name, 'unknown') AS repo_name,
                    COALESCE(SUM(estimated_cost_usd), 0)::numeric(14, 6) AS cost_usd,
                    COUNT(*)::bigint AS call_count
                FROM llm_call_metrics
                WHERE created_at >= $1::timestamptz - (
                    CASE $2
                        WHEN '24h' THEN INTERVAL '24 hours'
                        WHEN '7d' THEN INTERVAL '7 days'
                        WHEN '30d' THEN INTERVAL '30 days'
                    END
                )
                  AND created_at <= $1::timestamptz
                GROUP BY COALESCE(repo_name, 'unknown')
                ORDER BY cost_usd DESC, repo_name ASC
                """,
                timestamp,
                window,
            )
        snapshot = ModelCostByRepoSnapshot(
            window=window,  # type: ignore[arg-type]
            rows=[
                ModelCostByRepoSnapshotRow(
                    repo_name=str(_row_get(row, "repo_name")),
                    cost_usd=_decimal(_row_get(row, "cost_usd")),
                    call_count=_int(_row_get(row, "call_count")),
                )
                for row in rows
            ],
            snapshot_timestamp=timestamp,
        )
        payload = snapshot.model_dump(mode="json")
        store_latest_snapshot(TOPIC_COST_BY_REPO, window, payload)
        if publisher is not None:
            await publisher.publish(TOPIC_COST_BY_REPO, payload)
        return snapshot


__all__ = ["HandlerProjectionCostByRepo", "SnapshotPublisher"]
