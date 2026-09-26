# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A source-coordinate outbox preserves bytes and refuses replay collisions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from omnibase_infra.runtime.sim_archive_rehydration_outbox import (
    PostgresSimArchiveRehydrationOutbox,
)


class _Connection:
    def __init__(self) -> None:
        self.existing: dict[str, object] | None = None
        self.pending: list[dict[str, object]] = []
        self.inserted = True
        self.statements: list[str] = []

    @asynccontextmanager
    async def transaction(self) -> Any:
        yield self

    async def fetchrow(self, sql: str, *args: object) -> dict[str, object] | None:
        self.statements.append(sql)
        if "INSERT INTO" in sql:
            return {"source_topic": args[0]} if self.inserted else None
        return self.existing

    async def fetch(self, sql: str, *_args: object) -> list[dict[str, object]]:
        self.statements.append(sql)
        return self.pending

    async def execute(self, sql: str, *_args: object) -> str:
        self.statements.append(sql)
        return "UPDATE 1"


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self.connection


class _Publisher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    async def publish(
        self,
        topic: str,
        *,
        key: bytes | None,
        value: bytes,
        headers: list[tuple[str, bytes | None]],
        timestamp_ms: int,
    ) -> None:
        if self.fail:
            raise RuntimeError("broker unavailable")
        self.calls.append((topic, key, value, headers, timestamp_ms))


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        target_topic="onex.cmd.omnimarket.delegate-skill.v1",
        source_key=("onex.cmd.omnimarket.delegate-skill.v1", 2, 11),
        key=b"\x00key",
        value=b"\x00value",
        headers=(("original", b"\xff"), ("repeated", b"a"), ("repeated", b"b")),
        timestamp_ms=1_800_000_000_123,
    )


_TARGETS = frozenset({"onex.cmd.omnimarket.delegate-skill.v1"})


@pytest.fixture(autouse=True)
def _sim_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "sim-202")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_enqueue_is_atomic_and_same_source_collision_refuses() -> None:
    connection = _Connection()
    outbox = PostgresSimArchiveRehydrationOutbox(
        _Pool(connection), _Publisher(), rehydration_targets=_TARGETS
    )
    plan = _plan()

    assert await outbox.enqueue_once(plan) is True
    assert "ON CONFLICT" in connection.statements[0]
    connection.inserted = False
    connection.existing = {
        "target_topic": plan.target_topic,
        "record_key": plan.key,
        "record_value": plan.value,
        "headers_json": outbox.encode_headers(plan.headers),
        "timestamp_ms": plan.timestamp_ms,
    }
    assert await outbox.enqueue_once(plan) is False

    connection.existing["record_value"] = b"different"
    with pytest.raises(ValueError, match="collision"):
        await outbox.enqueue_once(plan)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relay_keeps_row_pending_until_broker_confirms() -> None:
    connection = _Connection()
    plan = _plan()
    outbox = PostgresSimArchiveRehydrationOutbox(
        _Pool(connection), _Publisher(), rehydration_targets=_TARGETS
    )
    connection.pending = [
        {
            "source_topic": plan.source_key[0],
            "source_partition": plan.source_key[1],
            "source_offset": plan.source_key[2],
            "target_topic": plan.target_topic,
            "record_key": plan.key,
            "record_value": plan.value,
            "headers_json": outbox.encode_headers(plan.headers),
            "timestamp_ms": plan.timestamp_ms,
        }
    ]
    failing = PostgresSimArchiveRehydrationOutbox(
        _Pool(connection), _Publisher(fail=True), rehydration_targets=_TARGETS
    )
    with pytest.raises(RuntimeError, match="broker unavailable"):
        await failing.relay_once()
    assert not any(sql.lstrip().startswith("UPDATE") for sql in connection.statements)

    publisher = _Publisher()
    working = PostgresSimArchiveRehydrationOutbox(
        _Pool(connection), publisher, rehydration_targets=_TARGETS
    )
    assert await working.relay_once() == 1
    assert publisher.calls == [
        (plan.target_topic, plan.key, plan.value, list(plan.headers), plan.timestamp_ms)
    ]
    assert any("FOR UPDATE SKIP LOCKED" in sql for sql in connection.statements)
    assert any(sql.lstrip().startswith("UPDATE") for sql in connection.statements)


@pytest.mark.unit
def test_outbox_rejects_non_sim_runtime_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dev-201")
    with pytest.raises(ValueError, match="sim-202"):
        PostgresSimArchiveRehydrationOutbox(
            _Pool(_Connection()), _Publisher(), rehydration_targets=_TARGETS
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_outbox_refuses_target_outside_verified_contract_allowlist() -> None:
    connection = _Connection()
    outbox = PostgresSimArchiveRehydrationOutbox(
        _Pool(connection), _Publisher(), rehydration_targets=_TARGETS
    )
    plan = _plan()
    plan.target_topic = "onex.cmd.other-topic.v1"
    plan.source_key = (plan.target_topic, 2, 11)
    with pytest.raises(ValueError, match="invalid sim archive"):
        await outbox.enqueue_once(plan)
    assert connection.statements == []


@pytest.mark.unit
def test_outbox_requires_nonempty_verified_targets() -> None:
    with pytest.raises(ValueError, match="rehydration_targets"):
        PostgresSimArchiveRehydrationOutbox(
            _Pool(_Connection()), _Publisher(), rehydration_targets=frozenset()
        )
    with pytest.raises(ValueError, match="rehydration_targets"):
        PostgresSimArchiveRehydrationOutbox(
            _Pool(_Connection()), _Publisher(), rehydration_targets=frozenset({" "})
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_relay_refuses_stored_target_outside_verified_allowlist() -> None:
    connection = _Connection()
    plan = _plan()
    publisher = _Publisher()
    outbox = PostgresSimArchiveRehydrationOutbox(
        _Pool(connection), publisher, rehydration_targets=_TARGETS
    )
    connection.pending = [
        {
            "source_topic": "onex.cmd.other-topic.v1",
            "source_partition": 2,
            "source_offset": 11,
            "target_topic": "onex.cmd.other-topic.v1",
            "record_key": plan.key,
            "record_value": plan.value,
            "headers_json": outbox.encode_headers(plan.headers),
            "timestamp_ms": plan.timestamp_ms,
        }
    ]
    with pytest.raises(ValueError, match="not allowed"):
        await outbox.relay_once()
    assert publisher.calls == []
    assert not any(sql.lstrip().startswith("UPDATE") for sql in connection.statements)
