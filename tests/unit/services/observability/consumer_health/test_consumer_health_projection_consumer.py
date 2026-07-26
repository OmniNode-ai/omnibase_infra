# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dedicated unit tests for ConsumerHealthProjectionConsumer (OMN-7904).

Covers the previously-untested projection consumer in
``omnibase_infra.services.observability.consumer_health.consumer``:

* ``_mask_dsn_password`` DSN redaction (port / no-port / no-password / malformed)
* ``EnumHealthStatus`` values
* ``_determine_health`` poll/write staleness classification
* ``_handle_health`` HTTP endpoint status + body
* ``health_check`` programmatic snapshot
* ``_flush_batch`` write/commit/failure/short-write accounting
* ``run`` main loop happy-path, timeout, and KafkaError branches
* ``start`` / ``stop`` lifecycle wiring

All Kafka / PostgreSQL / aiohttp boundaries are mocked in-memory; no real
infrastructure is touched.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

from omnibase_infra.services.observability.consumer_health import (
    consumer as consumer_mod,
)
from omnibase_infra.services.observability.consumer_health.config import (
    ConfigConsumerHealthProjection,
)
from omnibase_infra.services.observability.consumer_health.consumer import (
    ConsumerHealthProjectionConsumer,
    EnumHealthStatus,
    _mask_dsn_password,
)


def _make_config(**overrides: object) -> ConfigConsumerHealthProjection:
    """Build a valid config with all required fields, allowing overrides."""
    params: dict[str, object] = {
        "kafka_bootstrap_servers": "localhost:19092",
        "postgres_dsn": "postgresql://postgres:secret@localhost:5432/omnibase_infra",
    }
    params.update(overrides)
    return ConfigConsumerHealthProjection(**params)  # type: ignore[arg-type]


def _make_consumer(**overrides: object) -> ConsumerHealthProjectionConsumer:
    return ConsumerHealthProjectionConsumer(_make_config(**overrides))


# --------------------------------------------------------------------------- #
# _mask_dsn_password
# --------------------------------------------------------------------------- #


def test_mask_dsn_password_with_port() -> None:
    dsn = "postgresql://user:supersecret@db.example.com:5432/mydb"
    masked = _mask_dsn_password(dsn)
    assert "supersecret" not in masked
    assert "user:***@db.example.com:5432" in masked
    assert masked.startswith("postgresql://")
    assert masked.endswith("/mydb")


def test_mask_dsn_password_without_port() -> None:
    dsn = "postgresql://user:supersecret@db.example.com/mydb"
    masked = _mask_dsn_password(dsn)
    assert "supersecret" not in masked
    assert "user:***@db.example.com" in masked
    assert ":5432" not in masked


def test_mask_dsn_password_no_password_returned_unchanged() -> None:
    dsn = "postgresql://db.example.com:5432/mydb"
    assert _mask_dsn_password(dsn) == dsn


def test_mask_dsn_password_malformed_returns_redacted_fallback() -> None:
    # Invalid IPv6 literal makes urlparse raise ValueError -> redacted fallback.
    assert _mask_dsn_password("postgresql://user:pass@[::1/db") == "***redacted***"


# --------------------------------------------------------------------------- #
# EnumHealthStatus
# --------------------------------------------------------------------------- #


def test_enum_health_status_values() -> None:
    assert EnumHealthStatus.HEALTHY.value == "healthy"
    assert EnumHealthStatus.DEGRADED.value == "degraded"
    assert EnumHealthStatus.UNHEALTHY.value == "unhealthy"


# --------------------------------------------------------------------------- #
# _determine_health
# --------------------------------------------------------------------------- #


def test_determine_health_no_activity_is_healthy() -> None:
    consumer = _make_consumer()
    assert consumer._determine_health() == EnumHealthStatus.HEALTHY


def test_determine_health_stale_poll_is_unhealthy() -> None:
    consumer = _make_consumer(health_check_poll_staleness_seconds=60)
    consumer._last_poll_at = datetime.now(UTC) - timedelta(seconds=120)
    assert consumer._determine_health() == EnumHealthStatus.UNHEALTHY


def test_determine_health_stale_write_is_degraded() -> None:
    consumer = _make_consumer(
        health_check_poll_staleness_seconds=60,
        health_check_staleness_seconds=300,
    )
    now = datetime.now(UTC)
    consumer._last_poll_at = now  # fresh poll -> not UNHEALTHY
    consumer._last_write_at = now - timedelta(seconds=600)  # stale write
    assert consumer._determine_health() == EnumHealthStatus.DEGRADED


def test_determine_health_fresh_poll_and_write_is_healthy() -> None:
    consumer = _make_consumer()
    now = datetime.now(UTC)
    consumer._last_poll_at = now
    consumer._last_write_at = now
    assert consumer._determine_health() == EnumHealthStatus.HEALTHY


# --------------------------------------------------------------------------- #
# _handle_health / health_check
# --------------------------------------------------------------------------- #


async def test_handle_health_healthy_returns_200() -> None:
    consumer = _make_consumer()
    consumer._messages_received = 7
    consumer._messages_processed = 5
    consumer._messages_failed = 2
    consumer._batches_processed = 1
    consumer._started_at = datetime.now(UTC)

    response = await consumer._handle_health(MagicMock())

    assert response.status == 200
    assert isinstance(response.body, (bytes, bytearray))
    body = json.loads(response.body)
    assert body["status"] == "healthy"
    assert body["messages_received"] == 7
    assert body["messages_processed"] == 5
    assert body["messages_failed"] == 2
    assert body["batches_processed"] == 1
    assert body["consumer_group"] == consumer._config.kafka_group_id
    assert body["started_at"] is not None
    assert body["last_poll_at"] is None


async def test_handle_health_unhealthy_returns_503() -> None:
    consumer = _make_consumer(health_check_poll_staleness_seconds=60)
    consumer._last_poll_at = datetime.now(UTC) - timedelta(seconds=120)

    response = await consumer._handle_health(MagicMock())

    assert response.status == 503
    assert isinstance(response.body, (bytes, bytearray))
    body = json.loads(response.body)
    assert body["status"] == "unhealthy"
    assert body["last_poll_at"] is not None


async def test_health_check_programmatic_snapshot() -> None:
    consumer = _make_consumer()
    consumer._messages_received = 3
    consumer._messages_processed = 3
    consumer._messages_failed = 0

    snapshot = await consumer.health_check()

    assert snapshot["status"] == "healthy"
    assert snapshot["messages_received"] == 3
    assert snapshot["messages_processed"] == 3
    assert snapshot["messages_failed"] == 0


# --------------------------------------------------------------------------- #
# _flush_batch
# --------------------------------------------------------------------------- #


def _make_batch(
    events: list[dict[str, object]],
) -> list[tuple[object, dict[str, object]]]:
    return [(SimpleNamespace(value=event), event) for event in events]


async def test_flush_batch_empty_is_noop() -> None:
    consumer = _make_consumer()
    consumer._writer = MagicMock()
    consumer._consumer = AsyncMock()
    await consumer._flush_batch([])
    consumer._writer.write_batch.assert_not_called()


async def test_flush_batch_missing_writer_is_noop() -> None:
    consumer = _make_consumer()
    # writer is None by construction; consumer present
    consumer._consumer = AsyncMock()
    await consumer._flush_batch(_make_batch([{"event_id": "e1"}]))
    # No crash, no metrics moved.
    assert consumer._batches_processed == 0


async def test_flush_batch_success_updates_metrics_and_commits() -> None:
    consumer = _make_consumer()
    writer = MagicMock()
    writer.write_batch = AsyncMock(return_value=2)
    consumer._writer = writer
    kafka = AsyncMock(spec=AIOKafkaConsumer)
    consumer._consumer = kafka

    batch = _make_batch([{"event_id": "e1"}, {"event_id": "e2"}])
    await consumer._flush_batch(batch)

    writer.write_batch.assert_awaited_once()
    assert consumer._messages_processed == 2
    assert consumer._messages_failed == 0
    assert consumer._batches_processed == 1
    assert consumer._last_write_at is not None
    kafka.commit.assert_awaited_once()


async def test_flush_batch_short_write_counts_failures() -> None:
    consumer = _make_consumer()
    writer = MagicMock()
    writer.write_batch = AsyncMock(return_value=1)  # only 1 of 2 written
    consumer._writer = writer
    consumer._consumer = AsyncMock()

    await consumer._flush_batch(_make_batch([{"event_id": "e1"}, {"event_id": "e2"}]))

    assert consumer._messages_processed == 1
    assert consumer._messages_failed == 1


async def test_flush_batch_writer_exception_counts_all_failed() -> None:
    consumer = _make_consumer()
    writer = MagicMock()
    writer.write_batch = AsyncMock(side_effect=RuntimeError("db down"))
    consumer._writer = writer
    kafka = AsyncMock(spec=AIOKafkaConsumer)
    consumer._consumer = kafka

    # Must not raise — exception is swallowed and logged.
    await consumer._flush_batch(_make_batch([{"event_id": "e1"}, {"event_id": "e2"}]))

    assert consumer._messages_failed == 2
    assert consumer._batches_processed == 0
    kafka.commit.assert_not_called()


# --------------------------------------------------------------------------- #
# run() main loop
# --------------------------------------------------------------------------- #


async def test_run_processes_record_and_flushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _make_consumer(batch_size=1, batch_timeout_ms=100)
    record = SimpleNamespace(value={"event_id": "e1"})

    async def fake_getmany(**_kwargs: object) -> dict[tuple[str, int], list[object]]:
        consumer._shutdown_event.set()  # exit after this iteration
        return {("topic", 0): [record]}

    kafka = AsyncMock(spec=AIOKafkaConsumer)
    kafka.getmany = AsyncMock(side_effect=fake_getmany)
    consumer._consumer = kafka

    writer = MagicMock()
    writer.write_batch = AsyncMock(return_value=1)
    consumer._writer = writer

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *a, **k: None)

    await consumer.run()

    assert consumer._messages_received == 1
    assert consumer._messages_processed == 1
    writer.write_batch.assert_awaited_once()
    kafka.commit.assert_awaited_once()


async def test_run_skips_none_valued_records(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = _make_consumer(batch_size=1, batch_timeout_ms=100)
    record = SimpleNamespace(value=None)  # tombstone / non-JSON

    async def fake_getmany(**_kwargs: object) -> dict[tuple[str, int], list[object]]:
        consumer._shutdown_event.set()
        return {("topic", 0): [record]}

    kafka = AsyncMock(spec=AIOKafkaConsumer)
    kafka.getmany = AsyncMock(side_effect=fake_getmany)
    consumer._consumer = kafka
    writer = MagicMock()
    writer.write_batch = AsyncMock(return_value=0)
    consumer._writer = writer

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *a, **k: None)

    await consumer.run()

    # Received counts the record, but None value is not batched -> no write.
    assert consumer._messages_received == 1
    writer.write_batch.assert_not_called()


async def test_run_handles_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = _make_consumer()

    async def fake_getmany(**_kwargs: object) -> dict[tuple[str, int], list[object]]:
        consumer._shutdown_event.set()
        raise TimeoutError

    kafka = AsyncMock(spec=AIOKafkaConsumer)
    kafka.getmany = AsyncMock(side_effect=fake_getmany)
    consumer._consumer = kafka

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *a, **k: None)

    # Should exit cleanly without raising.
    await consumer.run()
    assert consumer._messages_received == 0


async def test_run_handles_kafka_error(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = _make_consumer()

    async def fake_getmany(**_kwargs: object) -> dict[tuple[str, int], list[object]]:
        consumer._shutdown_event.set()
        raise KafkaError("broker unreachable")

    kafka = AsyncMock(spec=AIOKafkaConsumer)
    kafka.getmany = AsyncMock(side_effect=fake_getmany)
    consumer._consumer = kafka

    sleep_mock = AsyncMock()
    monkeypatch.setattr(consumer_mod.asyncio, "sleep", sleep_mock)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *a, **k: None)

    await consumer.run()

    sleep_mock.assert_awaited()  # backoff sleep on Kafka error


# --------------------------------------------------------------------------- #
# start() / stop() lifecycle
# --------------------------------------------------------------------------- #


async def test_start_initializes_pool_consumer_and_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _make_consumer()

    mock_pool = AsyncMock()
    monkeypatch.setattr(
        consumer_mod.asyncpg, "create_pool", AsyncMock(return_value=mock_pool)
    )
    mock_kafka_instance = AsyncMock(spec=AIOKafkaConsumer)
    monkeypatch.setattr(
        consumer_mod, "AIOKafkaConsumer", MagicMock(return_value=mock_kafka_instance)
    )
    monkeypatch.setattr(consumer_mod, "WriterConsumerHealthPostgres", MagicMock())
    monkeypatch.setattr(consumer, "_start_health_check", AsyncMock())

    await consumer.start()

    assert consumer._pool is mock_pool
    assert consumer._writer is not None
    assert consumer._consumer is mock_kafka_instance
    mock_kafka_instance.start.assert_awaited_once()
    consumer._start_health_check.assert_awaited_once()  # type: ignore[attr-defined]
    assert consumer._started_at is not None


async def test_stop_closes_all_resources() -> None:
    consumer = _make_consumer()
    consumer._consumer = AsyncMock()
    consumer._pool = AsyncMock()
    consumer._health_runner = AsyncMock()

    await consumer.stop()

    consumer._consumer.stop.assert_awaited_once()
    consumer._pool.close.assert_awaited_once()
    consumer._health_runner.cleanup.assert_awaited_once()
    assert consumer._shutdown_event.is_set()


async def test_stop_is_safe_with_no_resources() -> None:
    consumer = _make_consumer()
    # All resources None by construction — must not raise.
    await consumer.stop()
    assert consumer._shutdown_event.is_set()
