# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Database handlers for LLM cost API routes."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from omnibase_infra.services.cost_api.models import (
    AggregationWindow,
    ModelCostBreakdown,
    ModelCostBreakdownItem,
    ModelCostSummary,
    ModelCostTrend,
    ModelCostTrendPoint,
    ModelTokenUsage,
    TrendBucket,
)

if TYPE_CHECKING:
    import asyncpg


class RowLookup(Protocol):
    """Minimal row interface shared by asyncpg.Record and test rows."""

    def __getitem__(self, key: str) -> object:
        """Return the row value for a string key."""
        raise NotImplementedError


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
    return int(value)  # type: ignore[call-overload]


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
    """Fetch totals from canonical session/composite-session aggregate rows."""
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
    """Fetch token totals from canonical session/composite-session aggregate rows."""
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
            """,
            window,
        )
    return ModelTokenUsage(
        window=window,
        total_tokens=_int(_row_get(row, "total_tokens")),
        call_count=_int(_row_get(row, "call_count")),
        average_tokens_per_call=_decimal(_row_get(row, "average_tokens_per_call")),
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
