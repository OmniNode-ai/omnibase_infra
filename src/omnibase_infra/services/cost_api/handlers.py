# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Database handlers for LLM cost API routes."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, SupportsInt, cast

from omnibase_infra.services.cost_api.models import (
    AggregationWindow,
    ModelCostBreakdown,
    ModelCostBreakdownItem,
    ModelCostSummary,
    ModelCostTrend,
    ModelCostTrendPoint,
    ModelSavingsSummary,
    ModelSavingsSummaryItem,
    ModelTokenUsage,
    TrendBucket,
)
from omnibase_infra.services.cost_api.snapshot_cache import (
    TOPIC_COST_BY_REPO,
    TOPIC_COST_SUMMARY,
    TOPIC_COST_TOKEN_USAGE,
    get_latest_snapshot,
)

if TYPE_CHECKING:
    import asyncpg


class RowLookup(Protocol):
    """Minimal row interface shared by asyncpg.Record and test rows."""

    def __getitem__(self, key: str) -> object: ...


def _decimal(value: object, default: str = "0.000000") -> Decimal:
    if value is None:
        return Decimal(default)
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str | bytes | bytearray):
        return int(value)
    return int(cast("SupportsInt", value))


def _row_get(row: RowLookup | None, key: str, default: object = None) -> object:
    if row is None:
        return default
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


async def fetch_cost_summary(
    pool: asyncpg.Pool,
    *,
    window: AggregationWindow,
) -> ModelCostSummary:
    """Fetch totals from canonical session aggregate rows."""
    snapshot = get_latest_snapshot(TOPIC_COST_SUMMARY, window)
    if snapshot is not None:
        return ModelCostSummary(
            window=window,
            total_cost_usd=_decimal(snapshot.get("total_cost_usd")),
            total_tokens=_int(snapshot.get("total_tokens")),
            call_count=_int(snapshot.get("session_count")),
            estimated_coverage_pct=None,
        )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(total_cost_usd), 0)::numeric(14, 6) AS total_cost_usd,
                COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
                COALESCE(SUM(call_count), 0)::bigint AS call_count,
                CASE
                    WHEN COALESCE(SUM(call_count), 0) = 0 THEN NULL
                    ELSE (
                        SUM(COALESCE(estimated_coverage_pct, 0) * call_count)
                        / NULLIF(SUM(call_count), 0)
                    )::numeric(5, 2)
                END AS estimated_coverage_pct
            FROM llm_cost_aggregates
            WHERE "window" = $1::cost_aggregation_window
              AND aggregation_key LIKE 'session:%'
              AND aggregation_key NOT LIKE 'session:%;%'
            """,
            window,
        )
    return ModelCostSummary(
        window=window,
        total_cost_usd=_decimal(_row_get(row, "total_cost_usd")),
        total_tokens=_int(_row_get(row, "total_tokens")),
        call_count=_int(_row_get(row, "call_count")),
        estimated_coverage_pct=(
            None
            if _row_get(row, "estimated_coverage_pct") is None
            else _decimal(_row_get(row, "estimated_coverage_pct"), "0.00")
        ),
    )


async def fetch_cost_trend(
    pool: asyncpg.Pool,
    *,
    bucket: TrendBucket,
    days: int,
) -> ModelCostTrend:
    """Fetch event-time trend buckets from raw LLM call metrics.

    ``llm_cost_aggregates`` stores rolling aggregate windows, so it is not a
    reliable source for time series buckets. Raw ``created_at`` timestamps are
    used here to avoid presenting rolling-window updates as historical events.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                date_trunc($1, created_at) AS bucket_start,
                COALESCE(SUM(estimated_cost_usd), 0)::numeric(14, 6) AS total_cost_usd,
                COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
                COUNT(*)::bigint AS call_count
            FROM llm_call_metrics
            WHERE created_at >= NOW() - ($2::int * INTERVAL '1 day')
            GROUP BY bucket_start
            ORDER BY bucket_start ASC
            """,
            bucket,
            days,
        )
    return ModelCostTrend(
        bucket=bucket,
        days=days,
        points=[
            ModelCostTrendPoint(
                bucket_start=str(_row_get(row, "bucket_start")),
                total_cost_usd=_decimal(_row_get(row, "total_cost_usd")),
                total_tokens=_int(_row_get(row, "total_tokens")),
                call_count=_int(_row_get(row, "call_count")),
            )
            for row in rows
        ],
    )


async def fetch_cost_by_model(
    pool: asyncpg.Pool,
    *,
    window: AggregationWindow,
) -> ModelCostBreakdown:
    """Fetch model cost groups from composite session keys or legacy model keys."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH composite AS (
                SELECT
                    split_part(split_part(aggregation_key, ';model:', 2), ';', 1) AS name,
                    SUM(total_cost_usd)::numeric(14, 6) AS total_cost_usd,
                    SUM(total_tokens)::bigint AS total_tokens,
                    SUM(call_count)::bigint AS call_count
                FROM llm_cost_aggregates
                WHERE "window" = $1::cost_aggregation_window
                  AND aggregation_key LIKE 'session:%'
                  AND aggregation_key LIKE '%;model:%'
                GROUP BY name
            ),
            legacy AS (
                SELECT
                    substring(aggregation_key FROM '^model:(.*)$') AS name,
                    SUM(total_cost_usd)::numeric(14, 6) AS total_cost_usd,
                    SUM(total_tokens)::bigint AS total_tokens,
                    SUM(call_count)::bigint AS call_count
                FROM llm_cost_aggregates
                WHERE "window" = $1::cost_aggregation_window
                  AND aggregation_key LIKE 'model:%'
                  AND NOT EXISTS (SELECT 1 FROM composite)
                GROUP BY name
            )
            SELECT * FROM composite
            UNION ALL
            SELECT * FROM legacy
            WHERE name IS NOT NULL
            ORDER BY total_cost_usd DESC, name ASC
            """,
            window,
        )
    return _breakdown_from_rows(window=window, rows=rows)


async def fetch_cost_by_repo(
    pool: asyncpg.Pool,
    *,
    window: AggregationWindow,
) -> ModelCostBreakdown:
    """Fetch repo cost groups from composite session keys or legacy repo keys."""
    snapshot = get_latest_snapshot(TOPIC_COST_BY_REPO, window)
    if snapshot is not None:
        raw_rows = snapshot.get("rows")
        rows = raw_rows if isinstance(raw_rows, list) else []
        return ModelCostBreakdown(
            window=window,
            items=[
                ModelCostBreakdownItem(
                    name=str(row.get("repo_name") or "unknown"),
                    total_cost_usd=_decimal(row.get("cost_usd")),
                    total_tokens=0,
                    call_count=_int(row.get("call_count")),
                )
                for row in rows
                if isinstance(row, dict)
            ],
        )

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH composite AS (
                SELECT
                    split_part(split_part(aggregation_key, ';repo:', 2), ';', 1) AS name,
                    SUM(total_cost_usd)::numeric(14, 6) AS total_cost_usd,
                    SUM(total_tokens)::bigint AS total_tokens,
                    SUM(call_count)::bigint AS call_count
                FROM llm_cost_aggregates
                WHERE "window" = $1::cost_aggregation_window
                  AND aggregation_key LIKE 'session:%'
                  AND aggregation_key LIKE '%;repo:%'
                GROUP BY name
            ),
            legacy AS (
                SELECT
                    substring(aggregation_key FROM '^repo:(.*)$') AS name,
                    SUM(total_cost_usd)::numeric(14, 6) AS total_cost_usd,
                    SUM(total_tokens)::bigint AS total_tokens,
                    SUM(call_count)::bigint AS call_count
                FROM llm_cost_aggregates
                WHERE "window" = $1::cost_aggregation_window
                  AND aggregation_key LIKE 'repo:%'
                  AND NOT EXISTS (SELECT 1 FROM composite)
                GROUP BY name
            )
            SELECT * FROM composite
            UNION ALL
            SELECT * FROM legacy
            WHERE name IS NOT NULL
            ORDER BY total_cost_usd DESC, name ASC
            """,
            window,
        )
    return _breakdown_from_rows(window=window, rows=rows)


async def fetch_token_usage(
    pool: asyncpg.Pool,
    *,
    window: AggregationWindow,
) -> ModelTokenUsage:
    """Fetch token totals from canonical session aggregate rows."""
    snapshot = get_latest_snapshot(TOPIC_COST_TOKEN_USAGE, window)
    if snapshot is not None:
        raw_rows = snapshot.get("rows")
        rows = raw_rows if isinstance(raw_rows, list) else []
        total_tokens = sum(
            _int(row.get("total_tokens")) for row in rows if isinstance(row, dict)
        )
        call_count = len([row for row in rows if isinstance(row, dict)])
        average = (
            Decimal(total_tokens) / Decimal(call_count)
            if call_count
            else Decimal("0.000000")
        )
        return ModelTokenUsage(
            window=window,
            total_tokens=total_tokens,
            call_count=call_count,
            average_tokens_per_call=average.quantize(Decimal("0.000001")),
        )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(total_tokens), 0)::bigint AS total_tokens,
                COALESCE(SUM(call_count), 0)::bigint AS call_count,
                CASE
                    WHEN COALESCE(SUM(call_count), 0) = 0 THEN 0
                    ELSE (
                        COALESCE(SUM(total_tokens), 0)::numeric
                        / NULLIF(SUM(call_count), 0)
                    )
                END::numeric(14, 6) AS average_tokens_per_call
            FROM llm_cost_aggregates
            WHERE "window" = $1::cost_aggregation_window
              AND aggregation_key LIKE 'session:%'
              AND aggregation_key NOT LIKE 'session:%;%'
            """,
            window,
        )
    return ModelTokenUsage(
        window=window,
        total_tokens=_int(_row_get(row, "total_tokens")),
        call_count=_int(_row_get(row, "call_count")),
        average_tokens_per_call=_decimal(_row_get(row, "average_tokens_per_call")),
    )


async def fetch_savings_summary(
    pool: asyncpg.Pool,
    *,
    window: AggregationWindow,
) -> ModelSavingsSummary:
    """Fetch savings totals for the requested trailing window."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(savings_usd), 0)::numeric(14, 6) AS total_savings_usd,
                COALESCE(SUM(local_cost_usd), 0)::numeric(14, 6) AS local_cost_usd,
                COALESCE(SUM(cloud_cost_usd), 0)::numeric(14, 6) AS cloud_cost_usd,
                COUNT(DISTINCT session_id)::bigint AS session_count
            FROM savings_estimates
            WHERE event_timestamp >= NOW() - (
                CASE $1
                    WHEN '24h' THEN INTERVAL '24 hours'
                    WHEN '7d' THEN INTERVAL '7 days'
                    WHEN '30d' THEN INTERVAL '30 days'
                END
            )
            """,
            window,
        )
        rows = await conn.fetch(
            """
            SELECT
                model_local,
                COALESCE(SUM(savings_usd), 0)::numeric(14, 6) AS total_savings_usd,
                COALESCE(SUM(local_cost_usd), 0)::numeric(14, 6) AS local_cost_usd,
                COALESCE(SUM(cloud_cost_usd), 0)::numeric(14, 6) AS cloud_cost_usd,
                COUNT(DISTINCT session_id)::bigint AS session_count
            FROM savings_estimates
            WHERE event_timestamp >= NOW() - (
                CASE $1
                    WHEN '24h' THEN INTERVAL '24 hours'
                    WHEN '7d' THEN INTERVAL '7 days'
                    WHEN '30d' THEN INTERVAL '30 days'
                END
            )
            GROUP BY model_local
            ORDER BY total_savings_usd DESC, model_local ASC
            """,
            window,
        )
    return ModelSavingsSummary(
        window=window,
        total_savings_usd=_decimal(_row_get(row, "total_savings_usd")),
        local_cost_usd=_decimal(_row_get(row, "local_cost_usd")),
        cloud_cost_usd=_decimal(_row_get(row, "cloud_cost_usd")),
        session_count=_int(_row_get(row, "session_count")),
        items=[
            ModelSavingsSummaryItem(
                model_local=str(_row_get(item, "model_local", "unknown")),
                total_savings_usd=_decimal(_row_get(item, "total_savings_usd")),
                local_cost_usd=_decimal(_row_get(item, "local_cost_usd")),
                cloud_cost_usd=_decimal(_row_get(item, "cloud_cost_usd")),
                session_count=_int(_row_get(item, "session_count")),
            )
            for item in rows
        ],
    )


def _breakdown_from_rows(
    *,
    window: AggregationWindow,
    rows: Sequence[RowLookup],
) -> ModelCostBreakdown:
    return ModelCostBreakdown(
        window=window,
        items=[
            ModelCostBreakdownItem(
                name=str(_row_get(row, "name", "unknown")),
                total_cost_usd=_decimal(_row_get(row, "total_cost_usd")),
                total_tokens=_int(_row_get(row, "total_tokens")),
                call_count=_int(_row_get(row, "call_count")),
            )
            for row in rows
        ],
    )
