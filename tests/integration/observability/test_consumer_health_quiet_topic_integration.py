# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A quiet topic after handled traffic keeps the consumer's /health green (OMN-19356).

Replays the sequence measured on the .201 dev lane on 2026-09-23 through the
real batch path and the real aiohttp ``/health`` handler of the skill-lifecycle
consumer: two events arrive, the real ``WriterSkillLifecyclePostgres`` drops
the whole batch at its schema filter without touching the database, and the
topic goes quiet. Before the fix the cumulative ``messages_received > 0`` rule
turned ``/health`` 503 once the last write was 300s old, and autoheal
restart-cycled the container.

The negative control drives a batch whose write fails against an unreachable
database: that traffic was never handled, so the same quiet period must still
read DEGRADED.

No broker and no database are needed: the pool stub is never reached on the
schema-skip path, and raises on the failing path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from aiokafka.structs import ConsumerRecord

from omnibase_infra.services.observability.skill_lifecycle import (
    consumer as consumer_module,
)
from omnibase_infra.services.observability.skill_lifecycle.config import (
    ConfigSkillLifecycleConsumer,
)
from omnibase_infra.services.observability.skill_lifecycle.consumer import (
    TOPIC_STARTED,
    SkillLifecycleConsumer,
)
from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
    WriterSkillLifecyclePostgres,
)

_START = datetime(2026, 9, 23, 21, 9, 14, tzinfo=UTC)


class _SettableClock(datetime):
    """A ``datetime`` whose ``now`` returns an instant the test sets."""

    instant: datetime = _START

    @classmethod
    def now(cls, tz: object = None) -> _SettableClock:
        return cls.instant  # type: ignore[return-value]


class _UnreachablePool:
    """An asyncpg pool stand-in whose every acquire fails like a dead database."""

    def acquire(self) -> object:
        raise ConnectionRefusedError("database unreachable (OMN-19356 control)")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> type[_SettableClock]:
    monkeypatch.setattr(consumer_module, "datetime", _SettableClock)
    _SettableClock.instant = _START
    return _SettableClock


def _record(offset: int, payload: dict[str, object]) -> ConsumerRecord[bytes, bytes]:
    value = json.dumps(payload).encode("utf-8")
    return ConsumerRecord(
        topic=TOPIC_STARTED,
        partition=0,
        offset=offset,
        timestamp=0,
        timestamp_type=0,
        key=None,
        value=value,
        checksum=None,
        serialized_key_size=0,
        serialized_value_size=len(value),
        headers=[],
    )


def _consumer(pool: object) -> SkillLifecycleConsumer:
    consumer = SkillLifecycleConsumer(
        ConfigSkillLifecycleConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://postgres:unused@localhost:5432/unused",
            dlq_enabled=False,
            _env_file=None,
        )
    )
    consumer._writer = WriterSkillLifecyclePostgres(pool)
    consumer._running = True
    return consumer


async def _health_status(consumer: SkillLifecycleConsumer) -> tuple[int, str]:
    app = web.Application()
    app.router.add_get("/health", consumer._health_handler)
    async with TestServer(app) as server:
        client: TestClient[web.Request, None] = TestClient(server)
        async with client:
            response = await client.get("/health")
            body = await response.json()
            return response.status, str(body["status"])


async def test_quiet_after_schema_skipped_batch_stays_healthy(
    clock: type[_SettableClock],
) -> None:
    consumer = _consumer(_UnreachablePool())
    await consumer.metrics.record_polled()
    await consumer._process_batch(
        [
            _record(0, {"event_id": "omn19356-1"}),
            _record(1, {"event_id": "omn19356-2"}),
        ]
    )

    # 21:16:10Z, the moment autoheal fired on the dev lane.
    clock.instant = _START + timedelta(seconds=416)
    await consumer.metrics.record_polled()

    assert await _health_status(consumer) == (200, "healthy")


async def test_quiet_after_failed_write_still_degrades(
    clock: type[_SettableClock],
) -> None:
    consumer = _consumer(_UnreachablePool())
    await consumer.metrics.record_polled()
    await consumer.metrics.record_processed(count=1)
    clock.instant = _START + timedelta(seconds=60)
    await consumer._process_batch(
        [
            _record(
                0,
                {
                    "event_id": "omn19356-3",
                    "run_id": "run-omn19356",
                    "skill_name": "proof-skill",
                    "repo_id": "omnibase_infra",
                    "correlation_id": "corr-omn19356",
                    "emitted_at": _START.isoformat(),
                },
            )
        ]
    )
    assert consumer.metrics.messages_failed == 1

    clock.instant = _START + timedelta(seconds=416)
    await consumer.metrics.record_polled()

    assert await _health_status(consumer) == (503, "degraded")
