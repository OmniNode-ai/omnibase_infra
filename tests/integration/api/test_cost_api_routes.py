# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ASGI coverage for OMN-10334 cost API routes."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.services.cost_api.snapshot_cache import (
    TOPIC_COST_BY_REPO,
    TOPIC_COST_SUMMARY,
    TOPIC_COST_TOKEN_USAGE,
    clear_latest_snapshots,
    store_latest_snapshot,
)
from omnibase_infra.services.registry_api.main import create_app

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class FakeAcquire:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeConnection:
        return self._conn

    async def __aexit__(self, *_exc_info: object) -> None:
        return None


class FakePool:
    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self._conn)


class FakeConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, sql: str, *args: object) -> Mapping[str, object]:
        self.queries.append((sql, args))
        if "FROM savings_estimates" in sql:
            return {
                "total_savings_usd": Decimal("5.000000"),
                "local_cost_usd": Decimal("0.500000"),
                "cloud_cost_usd": Decimal("5.500000"),
                "session_count": 1,
            }
        if "average_tokens_per_call" in sql:
            return {
                "total_tokens": 6000,
                "call_count": 3,
                "average_tokens_per_call": Decimal("2000.000000"),
            }
        return {
            "total_cost_usd": Decimal("1.234567"),
            "total_tokens": 6000,
            "call_count": 3,
            "estimated_coverage_pct": Decimal("33.33"),
        }

    async def fetch(self, sql: str, *args: object) -> list[Mapping[str, object]]:
        self.queries.append((sql, args))
        if "FROM savings_estimates" in sql:
            return [
                {
                    "model_local": "qwen3-coder-30b",
                    "total_savings_usd": Decimal("5.000000"),
                    "local_cost_usd": Decimal("0.500000"),
                    "cloud_cost_usd": Decimal("5.500000"),
                    "session_count": 1,
                }
            ]
        if "llm_call_metrics" in sql:
            return [
                {
                    "bucket_start": "2026-04-28 00:00:00+00:00",
                    "total_cost_usd": Decimal("0.750000"),
                    "total_tokens": 3000,
                    "call_count": 2,
                },
                {
                    "bucket_start": "2026-04-29 00:00:00+00:00",
                    "total_cost_usd": Decimal("0.484567"),
                    "total_tokens": 3000,
                    "call_count": 1,
                },
            ]
        if ";model:" in sql:
            return [
                {
                    "name": "gpt-4.1",
                    "total_cost_usd": Decimal("1.000000"),
                    "total_tokens": 4000,
                    "call_count": 2,
                },
                {
                    "name": "gpt-4.1-mini",
                    "total_cost_usd": Decimal("0.234567"),
                    "total_tokens": 2000,
                    "call_count": 1,
                },
            ]
        if ";repo:" in sql:
            return [
                {
                    "name": "omnibase_infra",
                    "total_cost_usd": Decimal("1.111111"),
                    "total_tokens": 5000,
                    "call_count": 2,
                },
                {
                    "name": "omnibase_core",
                    "total_cost_usd": Decimal("0.123456"),
                    "total_tokens": 1000,
                    "call_count": 1,
                },
            ]
        raise AssertionError(f"unexpected fetch SQL: {sql}")


@pytest.fixture
def fake_conn() -> FakeConnection:
    clear_latest_snapshots()
    return FakeConnection()


@pytest.fixture
def app(fake_conn: FakeConnection) -> Any:
    return create_app(
        container=ModelONEXContainer(),
        cors_origins=["http://testserver"],
        cost_api_pool=FakePool(fake_conn),
    )


async def get_json(app: Any, path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(path)
    assert response.status_code == 200
    return response.json()


async def test_cost_summary_uses_canonical_session_totals(
    app: Any,
    fake_conn: FakeConnection,
) -> None:
    body = await get_json(app, "/api/costs/summary")

    assert body == {
        "window": "24h",
        "total_cost_usd": "1.234567",
        "total_tokens": 6000,
        "call_count": 3,
        "estimated_coverage_pct": "33.33",
    }
    sql, args = fake_conn.queries[-1]
    assert "aggregation_key LIKE 'session:%'" in sql
    assert "aggregation_key NOT LIKE 'session:%;%'" in sql
    assert args == ("24h",)


async def test_cost_trend_uses_raw_call_metric_timestamps(
    app: Any,
    fake_conn: FakeConnection,
) -> None:
    body = await get_json(app, "/api/costs/trend?bucket=day&days=7")

    assert body == {
        "bucket": "day",
        "days": 7,
        "points": [
            {
                "bucket_start": "2026-04-28 00:00:00+00:00",
                "total_cost_usd": "0.750000",
                "total_tokens": 3000,
                "call_count": 2,
            },
            {
                "bucket_start": "2026-04-29 00:00:00+00:00",
                "total_cost_usd": "0.484567",
                "total_tokens": 3000,
                "call_count": 1,
            },
        ],
    }
    trend_sql, trend_args = fake_conn.queries[-1]
    assert "FROM llm_call_metrics" in trend_sql
    assert "created_at" in trend_sql
    assert trend_args == ("day", 7)


async def test_cost_by_model_groups_composite_and_legacy_fallback(
    app: Any,
    fake_conn: FakeConnection,
) -> None:
    body = await get_json(app, "/api/costs/by-model")

    assert body == {
        "window": "24h",
        "items": [
            {
                "name": "gpt-4.1",
                "total_cost_usd": "1.000000",
                "total_tokens": 4000,
                "call_count": 2,
            },
            {
                "name": "gpt-4.1-mini",
                "total_cost_usd": "0.234567",
                "total_tokens": 2000,
                "call_count": 1,
            },
        ],
    }
    sql, args = fake_conn.queries[-1]
    assert "aggregation_key LIKE 'session:%'" in sql
    assert "aggregation_key LIKE 'model:%'" in sql
    assert "NOT EXISTS (SELECT 1 FROM composite)" in sql
    assert args == ("24h",)


async def test_cost_by_repo_groups_composite_and_legacy_fallback(
    app: Any,
    fake_conn: FakeConnection,
) -> None:
    body = await get_json(app, "/api/costs/by-repo")

    assert body == {
        "window": "24h",
        "items": [
            {
                "name": "omnibase_infra",
                "total_cost_usd": "1.111111",
                "total_tokens": 5000,
                "call_count": 2,
            },
            {
                "name": "omnibase_core",
                "total_cost_usd": "0.123456",
                "total_tokens": 1000,
                "call_count": 1,
            },
        ],
    }
    sql, args = fake_conn.queries[-1]
    assert "aggregation_key LIKE 'session:%'" in sql
    assert "aggregation_key LIKE 'repo:%'" in sql
    assert "NOT EXISTS (SELECT 1 FROM composite)" in sql
    assert args == ("24h",)


async def test_token_usage_response(app: Any, fake_conn: FakeConnection) -> None:
    body = await get_json(app, "/api/costs/token-usage")

    assert body == {
        "window": "24h",
        "total_tokens": 6000,
        "call_count": 3,
        "average_tokens_per_call": "2000.000000",
    }
    sql, args = fake_conn.queries[-1]
    assert "aggregation_key LIKE 'session:%'" in sql
    assert "aggregation_key NOT LIKE 'session:%;%'" in sql
    assert args == ("24h",)


async def test_cost_summary_prefers_latest_projection_snapshot(app: Any) -> None:
    store_latest_snapshot(
        TOPIC_COST_SUMMARY,
        "24h",
        {
            "window": "24h",
            "total_cost_usd": "9.000000",
            "total_savings_usd": "3.500000",
            "total_tokens": 9000,
            "session_count": 4,
            "snapshot_timestamp": "2026-04-29T12:00:00Z",
        },
    )

    body = await get_json(app, "/api/costs/summary")

    assert body == {
        "window": "24h",
        "total_cost_usd": "9.000000",
        "total_tokens": 9000,
        "call_count": 4,
        "estimated_coverage_pct": None,
    }


async def test_cost_by_repo_prefers_latest_projection_snapshot(app: Any) -> None:
    store_latest_snapshot(
        TOPIC_COST_BY_REPO,
        "24h",
        {
            "window": "24h",
            "rows": [
                {
                    "repo_name": "omnibase_core",
                    "cost_usd": "0.750000",
                    "call_count": 1,
                },
                {
                    "repo_name": "unknown",
                    "cost_usd": "0.100000",
                    "call_count": 1,
                },
            ],
            "snapshot_timestamp": "2026-04-29T12:00:00Z",
        },
    )

    body = await get_json(app, "/api/costs/by-repo")

    assert body == {
        "window": "24h",
        "items": [
            {
                "name": "omnibase_core",
                "total_cost_usd": "0.750000",
                "total_tokens": 0,
                "call_count": 1,
            },
            {
                "name": "unknown",
                "total_cost_usd": "0.100000",
                "total_tokens": 0,
                "call_count": 1,
            },
        ],
    }


async def test_token_usage_prefers_latest_projection_snapshot(app: Any) -> None:
    store_latest_snapshot(
        TOPIC_COST_TOKEN_USAGE,
        "24h",
        {
            "window": "24h",
            "rows": [
                {
                    "bucket_timestamp": "2026-04-29T10:00:00Z",
                    "model_id": "gpt-4.1",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                {
                    "bucket_timestamp": "2026-04-29T11:00:00Z",
                    "model_id": "gpt-4.1-mini",
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
            ],
            "snapshot_timestamp": "2026-04-29T12:00:00Z",
        },
    )

    body = await get_json(app, "/api/costs/token-usage")

    assert body == {
        "window": "24h",
        "total_tokens": 450,
        "call_count": 2,
        "average_tokens_per_call": "225.000000",
    }


async def test_savings_summary_route_returns_seeded_value(
    app: Any,
    fake_conn: FakeConnection,
) -> None:
    body = await get_json(app, "/api/savings/summary?window=7d")

    assert body == {
        "window": "7d",
        "total_savings_usd": "5.000000",
        "local_cost_usd": "0.500000",
        "cloud_cost_usd": "5.500000",
        "session_count": 1,
        "items": [
            {
                "model_local": "qwen3-coder-30b",
                "total_savings_usd": "5.000000",
                "local_cost_usd": "0.500000",
                "cloud_cost_usd": "5.500000",
                "session_count": 1,
            }
        ],
    }
    assert len(fake_conn.queries) >= 2
    summary_sql, summary_args = fake_conn.queries[-2]
    grouped_sql, grouped_args = fake_conn.queries[-1]
    assert "FROM savings_estimates" in summary_sql
    assert "FROM savings_estimates" in grouped_sql
    assert "GROUP BY model_local" in grouped_sql
    assert summary_args == ("7d",)
    assert grouped_args == ("7d",)


async def test_cost_pool_unavailable_response_carries_correlation_context() -> None:
    app = create_app(
        container=ModelONEXContainer(),
        cors_origins=["http://testserver"],
        cost_api_pool=None,
    )
    correlation_id = uuid4()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/costs/summary",
            headers={"X-Correlation-ID": str(correlation_id)},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["message"] == "Cost API database pool not configured"
    context = detail["context"]
    assert UUID(context["correlation_id"]) == correlation_id
    assert context["operation"] == "cost_api_pool_lookup"
    assert context["transport_type"] == "db"


async def test_openapi_registers_cost_and_savings_paths(app: Any) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/costs/summary" in paths
    assert "/api/costs/trend" in paths
    assert "/api/costs/by-model" in paths
    assert "/api/costs/by-repo" in paths
    assert "/api/costs/token-usage" in paths
    assert "/api/savings/summary" in paths
