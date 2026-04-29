# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Proof-of-life e2e coverage for cost observability snapshot topics."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, Mapping
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

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
from omnibase_infra.services.cost_api.snapshot_cache import (
    TOPIC_COST_BY_REPO,
    TOPIC_COST_SUMMARY,
    TOPIC_COST_TOKEN_USAGE,
    clear_latest_snapshots,
)
from tests.integration.nodes.test_node_projection_cost_snapshots import (
    FakeConnection,
    FakePool,
    _load_fixture,
    _load_golden,
)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_AVAILABLE = (
    KAFKA_BOOTSTRAP_SERVERS is not None
    and bool(KAFKA_BOOTSTRAP_SERVERS.strip())
    and os.getenv("KAFKA_INTEGRATION_TESTS") == "1"
)

SNAPSHOT_TOPICS = (
    TOPIC_COST_SUMMARY,
    TOPIC_COST_BY_REPO,
    TOPIC_COST_TOKEN_USAGE,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.kafka,
    pytest.mark.postgres,
    pytest.mark.skipif(
        not KAFKA_AVAILABLE,
        reason=(
            "Kafka e2e not enabled; set KAFKA_BOOTSTRAP_SERVERS and "
            "KAFKA_INTEGRATION_TESTS=1"
        ),
    ),
]


class KafkaSnapshotPublisher:
    """SnapshotPublisher adapter backed by an already-started Kafka producer."""

    def __init__(self, producer: object) -> None:
        self._producer = producer

    async def publish(self, topic: str, payload: dict[str, object]) -> object:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return await self._producer.send_and_wait(topic, encoded)  # type: ignore[attr-defined]


@pytest.fixture
def kafka_bootstrap_servers() -> str:
    assert KAFKA_BOOTSTRAP_SERVERS is not None
    return KAFKA_BOOTSTRAP_SERVERS


@pytest.fixture
async def kafka_clients(
    kafka_bootstrap_servers: str,
) -> AsyncGenerator[tuple[object, object], None]:
    """Start Kafka consumer before producer-triggered snapshot emission."""
    aiokafka = pytest.importorskip("aiokafka")

    group_id = f"e2e-omn-10344-cost-observability-{uuid4().hex}"
    consumer = aiokafka.AIOKafkaConsumer(
        bootstrap_servers=kafka_bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )
    producer = aiokafka.AIOKafkaProducer(
        bootstrap_servers=kafka_bootstrap_servers,
        acks="all",
    )

    await consumer.start()
    await producer.start()
    try:
        consumer.subscribe(SNAPSHOT_TOPICS)
        await _wait_for_assignment(consumer)
        yield producer, consumer
    finally:
        await producer.stop()
        await consumer.stop()


async def _wait_for_assignment(consumer: object) -> None:
    for _ in range(20):
        assignment = consumer.assignment()  # type: ignore[attr-defined]
        if assignment:
            return
        await consumer.getmany(timeout_ms=250)  # type: ignore[attr-defined]
    raise AssertionError("Kafka consumer did not receive topic assignment")


async def _read_snapshot_payloads(
    consumer: object,
    *,
    test_start: datetime,
) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for _ in range(30):
        batches = await consumer.getmany(timeout_ms=500)  # type: ignore[attr-defined]
        for messages in batches.values():
            for message in messages:
                topic = str(message.topic)
                payload = _decode_payload(bytes(message.value))
                if topic in SNAPSHOT_TOPICS and _matches_expected(topic, payload):
                    snapshot_timestamp = _parse_datetime(payload["snapshot_timestamp"])
                    if snapshot_timestamp >= test_start:
                        found[topic] = payload
        if set(found) == set(SNAPSHOT_TOPICS):
            return found
    raise AssertionError(
        f"missing snapshot topics: {sorted(set(SNAPSHOT_TOPICS) - set(found))}"
    )


def _decode_payload(value: bytes) -> dict[str, object]:
    payload = json.loads(value.decode("utf-8"))
    assert isinstance(payload, dict)
    return cast("dict[str, object]", payload)


def _matches_expected(topic: str, payload: Mapping[str, object]) -> bool:
    if payload.get("window") != "24h" or "snapshot_timestamp" not in payload:
        return False
    if topic == TOPIC_COST_SUMMARY:
        return payload.get("total_tokens") == 600 and payload.get("session_count") == 4
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return False
    if topic == TOPIC_COST_BY_REPO:
        repo_names = {row.get("repo_name") for row in rows if isinstance(row, dict)}
        return {"omnibase_core", "omnibase_infra", "unknown"} <= repo_names
    if topic == TOPIC_COST_TOKEN_USAGE:
        model_ids = {row.get("model_id") for row in rows if isinstance(row, dict)}
        return {"gpt-4.1", "gpt-4.1-mini"} <= model_ids
    return False


def _parse_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _expected_payloads(snapshot_timestamp: datetime) -> dict[str, dict[str, object]]:
    expected = {
        TOPIC_COST_SUMMARY: _load_golden("task-11-summary.golden.json"),
        TOPIC_COST_BY_REPO: _load_golden("task-11-by-repo.golden.json"),
        TOPIC_COST_TOKEN_USAGE: _load_golden("task-11-token-usage.golden.json"),
    }
    timestamp = snapshot_timestamp.isoformat().replace("+00:00", "Z")
    for payload in expected.values():
        payload["snapshot_timestamp"] = timestamp
    return expected


async def test_cost_snapshot_topics_emit_fixture_exact_payloads(
    kafka_clients: tuple[object, object],
) -> None:
    """Publish all cost snapshots to Kafka and consume exact fixture payloads."""
    clear_latest_snapshots()
    producer, consumer = kafka_clients
    publisher = KafkaSnapshotPublisher(producer)
    test_start = datetime.now(tz=UTC)
    snapshot_timestamp = datetime.now(tz=UTC)

    await HandlerProjectionCostSummary().emit_snapshot(
        FakePool(FakeConnection(_load_fixture("task-11-summary.fixtures.jsonl"))),
        publisher,
        window="24h",
        snapshot_timestamp=snapshot_timestamp,
    )
    await HandlerProjectionCostByRepo().emit_snapshot(
        FakePool(FakeConnection(_load_fixture("task-11-by-repo.fixtures.jsonl"))),
        publisher,
        window="24h",
        snapshot_timestamp=snapshot_timestamp,
    )
    await HandlerProjectionCostTokenUsage().emit_snapshot(
        FakePool(FakeConnection(_load_fixture("task-11-token-usage.fixtures.jsonl"))),
        publisher,
        window="24h",
        snapshot_timestamp=snapshot_timestamp,
    )

    actual = await _read_snapshot_payloads(consumer, test_start=test_start)

    assert actual == _expected_payloads(snapshot_timestamp)
    assert actual[TOPIC_COST_SUMMARY]["total_cost_usd"] == "1.600000"
    assert actual[TOPIC_COST_SUMMARY]["total_savings_usd"] == "3.500000"
    assert actual[TOPIC_COST_BY_REPO]["rows"] == [
        {"repo_name": "omnibase_core", "cost_usd": "0.750000", "call_count": 1},
        {"repo_name": "omnibase_infra", "cost_usd": "0.750000", "call_count": 2},
        {"repo_name": "unknown", "cost_usd": "0.100000", "call_count": 1},
    ]
    assert actual[TOPIC_COST_TOKEN_USAGE]["rows"] == [
        {
            "bucket_timestamp": "2026-04-29T09:00:00Z",
            "model_id": "gpt-4.1",
            "prompt_tokens": 30,
            "completion_tokens": 20,
            "total_tokens": 50,
        },
        {
            "bucket_timestamp": "2026-04-29T10:00:00Z",
            "model_id": "gpt-4.1",
            "prompt_tokens": 180,
            "completion_tokens": 70,
            "total_tokens": 250,
        },
        {
            "bucket_timestamp": "2026-04-29T11:00:00Z",
            "model_id": "gpt-4.1-mini",
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        },
    ]
