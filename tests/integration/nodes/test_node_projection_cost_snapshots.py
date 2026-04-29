# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay tests for Task 11 cost projection snapshot emitters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_projection_cost_by_repo.handlers.handler_projection_cost_by_repo import (
    HandlerProjectionCostByRepo,
)
from omnibase_infra.nodes.node_projection_cost_summary.handlers.handler_projection_cost_summary import (
    HandlerProjectionCostSummary,
)
from omnibase_infra.nodes.node_projection_cost_token_usage.handlers.handler_projection_cost_token_usage import (
    HandlerProjectionCostTokenUsage,
)
from omnibase_infra.services.cost_api.snapshot_cache import clear_latest_snapshots

pytestmark = pytest.mark.asyncio

SNAPSHOT_TS = datetime(2026, 4, 29, 12, 0, tzinfo=UTC)
FIXTURE_DIR = Path("tests/fixtures/cost_observability")


class FakePublisher:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object]]] = []

    async def publish(self, topic: str, payload: dict[str, object]) -> None:
        self.published.append((topic, payload))


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
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.llm_calls = [record for record in records if record["kind"] == "llm_call"]
        self.savings = [record for record in records if record["kind"] == "savings"]

    async def fetchrow(self, sql: str, *args: object) -> Mapping[str, object]:
        timestamp = (
            _as_datetime(args[0])
            if args and isinstance(args[0], datetime)
            else SNAPSHOT_TS
        )
        window = str(args[1]) if len(args) > 1 else str(args[0])
        if "FROM savings_estimates" in sql:
            rows = [
                row
                for row in self.savings
                if _within_window(
                    _as_datetime(row["event_timestamp"]), timestamp, window
                )
            ]
            return {
                "total_savings_usd": sum(
                    (_decimal(row["savings_usd"]) for row in rows),
                    Decimal("0.000000"),
                )
            }
        rows = list(self.llm_calls)
        return {
            "total_cost_usd": sum(
                (_decimal(row["estimated_cost_usd"]) for row in rows),
                Decimal("0.000000"),
            ),
            "total_tokens": sum(int(row["total_tokens"]) for row in rows),
            "call_count": len(rows),
        }

    async def fetch(self, sql: str, *args: object) -> list[Mapping[str, object]]:
        timestamp = _as_datetime(args[0])
        window = str(args[1])
        rows = [
            row
            for row in self.llm_calls
            if _within_window(_as_datetime(row["created_at"]), timestamp, window)
        ]
        if "COALESCE(repo_name, 'unknown')" in sql:
            grouped: dict[str, dict[str, object]] = {}
            for row in rows:
                repo = str(row["repo_name"] or "unknown")
                bucket = grouped.setdefault(
                    repo,
                    {
                        "repo_name": repo,
                        "cost_usd": Decimal("0.000000"),
                        "call_count": 0,
                    },
                )
                bucket["cost_usd"] = _decimal(bucket["cost_usd"]) + _decimal(
                    row["estimated_cost_usd"]
                )
                bucket["call_count"] = int(bucket["call_count"]) + 1
            return sorted(
                grouped.values(),
                key=lambda row: (-_decimal(row["cost_usd"]), str(row["repo_name"])),
            )

        bucket_size = str(args[2])
        grouped_tokens: dict[tuple[datetime, str], dict[str, object]] = {}
        for row in rows:
            bucket_ts = _truncate_bucket(_as_datetime(row["created_at"]), bucket_size)
            key = (bucket_ts, str(row["model_id"]))
            bucket = grouped_tokens.setdefault(
                key,
                {
                    "bucket_timestamp": bucket_ts,
                    "model_id": str(row["model_id"]),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + int(
                row["prompt_tokens"]
            )
            bucket["completion_tokens"] = int(bucket["completion_tokens"]) + int(
                row["completion_tokens"]
            )
            bucket["total_tokens"] = int(bucket["total_tokens"]) + int(
                row["total_tokens"]
            )
        return sorted(
            grouped_tokens.values(),
            key=lambda row: (row["bucket_timestamp"], str(row["model_id"])),
        )


def _load_fixture(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (FIXTURE_DIR / name).read_text(encoding="utf-8").splitlines()
    ]


def _load_golden(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _within_window(value: datetime, timestamp: datetime, window: str) -> bool:
    widths = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    return timestamp - widths[window] <= value <= timestamp


def _truncate_bucket(value: datetime, bucket: str) -> datetime:
    if bucket == "hour":
        return value.replace(minute=0, second=0, microsecond=0)
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


async def test_cost_summary_replay_matches_golden_field_by_field() -> None:
    clear_latest_snapshots()
    pool = FakePool(FakeConnection(_load_fixture("task-11-summary.fixtures.jsonl")))
    publisher = FakePublisher()

    snapshot = await HandlerProjectionCostSummary().emit_snapshot(
        pool,
        publisher,
        window="24h",
        snapshot_timestamp=SNAPSHOT_TS,
    )
    payload = snapshot.model_dump(mode="json")

    assert payload == _load_golden("task-11-summary.golden.json")
    assert snapshot.total_cost_usd == Decimal("1.600000")
    assert snapshot.total_savings_usd == Decimal("3.500000")
    assert snapshot.total_tokens == 600
    assert snapshot.session_count == 4
    assert publisher.published == [
        ("onex.snapshot.projection.cost.summary.v1", payload)
    ]


async def test_cost_by_repo_replay_matches_golden_field_by_field() -> None:
    clear_latest_snapshots()
    pool = FakePool(FakeConnection(_load_fixture("task-11-by-repo.fixtures.jsonl")))
    publisher = FakePublisher()

    snapshot = await HandlerProjectionCostByRepo().emit_snapshot(
        pool,
        publisher,
        window="24h",
        snapshot_timestamp=SNAPSHOT_TS,
    )
    payload = snapshot.model_dump(mode="json")

    assert payload == _load_golden("task-11-by-repo.golden.json")
    assert [row.repo_name for row in snapshot.rows] == [
        "omnibase_core",
        "omnibase_infra",
        "unknown",
    ]
    assert [row.cost_usd for row in snapshot.rows] == [
        Decimal("0.750000"),
        Decimal("0.750000"),
        Decimal("0.100000"),
    ]
    assert [row.call_count for row in snapshot.rows] == [1, 2, 1]
    assert publisher.published == [
        ("onex.snapshot.projection.cost.by_repo.v1", payload)
    ]


async def test_cost_token_usage_replay_matches_golden_field_by_field() -> None:
    clear_latest_snapshots()
    pool = FakePool(FakeConnection(_load_fixture("task-11-token-usage.fixtures.jsonl")))
    publisher = FakePublisher()

    snapshot = await HandlerProjectionCostTokenUsage().emit_snapshot(
        pool,
        publisher,
        window="24h",
        snapshot_timestamp=SNAPSHOT_TS,
    )
    payload = snapshot.model_dump(mode="json")

    assert payload == _load_golden("task-11-token-usage.golden.json")
    assert [row.model_id for row in snapshot.rows] == [
        "gpt-4.1",
        "gpt-4.1",
        "gpt-4.1-mini",
    ]
    assert [row.prompt_tokens for row in snapshot.rows] == [30, 180, 200]
    assert [row.completion_tokens for row in snapshot.rows] == [20, 70, 100]
    assert [row.total_tokens for row in snapshot.rows] == [50, 250, 300]
    assert publisher.published == [
        ("onex.snapshot.projection.cost.token_usage.v1", payload)
    ]
