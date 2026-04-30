# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for repeatable token usage snapshot emission."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.cost_projection_models import (
    ModelCostTokenUsageSnapshot,
    ModelCostTokenUsageSnapshotRow,
)
from omnibase_infra.services.cost_api.snapshot_cache import (
    TOPIC_COST_TOKEN_USAGE,
    store_latest_snapshot,
)


class SnapshotPublisher(Protocol):
    async def publish(self, topic: str, payload: dict[str, object]) -> object:
        raise NotImplementedError


def _int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _row_get(row: Mapping[str, object], key: str) -> object:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


class HandlerProjectionCostTokenUsage:
    """Compute and publish model/time-bucket token snapshots."""

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
    ) -> ModelCostTokenUsageSnapshot:
        """Compute one token usage snapshot and optionally publish it."""
        timestamp = snapshot_timestamp or datetime.now(tz=UTC)
        bucket = "hour" if window == "24h" else "day"
        async with pool.acquire() as conn:  # type: ignore[attr-defined]
            rows = await conn.fetch(
                """
                SELECT
                    date_trunc($3, created_at) AS bucket_timestamp,
                    model_id,
                    COALESCE(SUM(prompt_tokens), 0)::bigint AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0)::bigint AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
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
                GROUP BY bucket_timestamp, model_id
                ORDER BY bucket_timestamp ASC, model_id ASC
                """,
                timestamp,
                window,
                bucket,
            )
        snapshot = ModelCostTokenUsageSnapshot(
            window=window,  # type: ignore[arg-type]
            rows=[
                ModelCostTokenUsageSnapshotRow(
                    bucket_timestamp=_datetime(_row_get(row, "bucket_timestamp")),
                    provider_model_key=str(_row_get(row, "model_id")),
                    prompt_tokens=_int(_row_get(row, "prompt_tokens")),
                    completion_tokens=_int(_row_get(row, "completion_tokens")),
                    total_tokens=_int(_row_get(row, "total_tokens")),
                    call_count=_int(_row_get(row, "call_count")),
                )
                for row in rows
            ],
            snapshot_timestamp=timestamp,
        )
        payload = snapshot.model_dump(mode="json")
        store_latest_snapshot(TOPIC_COST_TOKEN_USAGE, window, payload)
        if publisher is not None:
            await publisher.publish(TOPIC_COST_TOKEN_USAGE, payload)
        return snapshot


__all__ = ["HandlerProjectionCostTokenUsage", "SnapshotPublisher"]
