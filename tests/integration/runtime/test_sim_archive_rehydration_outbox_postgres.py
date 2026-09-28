# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prove migration 108 and selected replay semantics against throwaway Postgres."""

from __future__ import annotations

from pathlib import Path

import asyncpg
import pytest

from omnibase_infra.runtime.sim_archive_rehydration_outbox import (
    PostgresSimArchiveRehydrationOutbox,
)
from omnibase_infra.runtime.sim_archive_source_receipt import (
    _RECEIPT_MINT,
    RawSimArchiveRecord,
    VerifiedSimArchiveRehydrationPlan,
)
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/108_create_sim_archive_rehydration_outbox.sql"
)


class _Broker:
    def __init__(self) -> None:
        self.fail = False
        self.acked: list[tuple[object, ...]] = []

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
            raise RuntimeError("broker ack unavailable")
        self.acked.append((topic, key, value, headers, timestamp_ms))


def _plan(offset: int) -> VerifiedSimArchiveRehydrationPlan:
    return VerifiedSimArchiveRehydrationPlan(
        RawSimArchiveRecord(
            topic=_TOPIC,
            partition=2,
            offset=offset,
            key=b"\x00\xffkey",
            value=b"\xff\x00payload",
            headers=(("same", b"\x00"), ("same", b"\xff"), ("empty", None)),
            timestamp_ms=1_800_000_000_123,
        ),
        _mint=_RECEIPT_MINT,
    )


async def test_real_postgres_outbox_selected_ack_and_collision(
    ephemeral_postgres: EphemeralPostgres,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "sim-202")
    applied = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(_MIGRATION))
    assert applied.returncode == 0, applied.stderr

    pool = await asyncpg.create_pool(
        host=ephemeral_postgres.socket_dir,
        port=ephemeral_postgres.port,
        user="postgres",
        database="postgres",
        min_size=1,
        max_size=2,
    )
    try:
        broker = _Broker()
        outbox = PostgresSimArchiveRehydrationOutbox(
            pool, broker, rehydration_targets=frozenset({_TOPIC})
        )
        selected = _plan(11)
        unrelated = _plan(12)

        await outbox.require_only_selected_keys((selected,))
        assert await outbox.enqueue_once(selected) is True
        assert await outbox.enqueue_once(selected) is False
        async with pool.acquire() as connection:
            stored = await connection.fetchrow(
                "SELECT record_key, record_value, headers_json, timestamp_ms "
                "FROM public.sim_archive_rehydration_outbox "
                "WHERE source_topic = $1 AND source_partition = $2 "
                "AND source_offset = $3",
                *selected.source_key,
            )
        assert stored is not None
        assert stored["record_key"] == selected.key
        assert stored["record_value"] == selected.value
        assert outbox.decode_headers(stored["headers_json"]) == list(selected.headers)
        assert stored["timestamp_ms"] == selected.timestamp_ms

        changed = VerifiedSimArchiveRehydrationPlan(
            RawSimArchiveRecord(
                topic=_TOPIC,
                partition=2,
                offset=11,
                key=selected.key,
                value=b"different",
                headers=selected.headers,
                timestamp_ms=selected.timestamp_ms,
            ),
            _mint=_RECEIPT_MINT,
        )
        with pytest.raises(ValueError, match="coordinate collision"):
            await outbox.enqueue_once(changed)

        broker.fail = True
        with pytest.raises(RuntimeError, match="broker ack unavailable"):
            await outbox.relay_selected_once((selected,))
        async with pool.acquire() as connection:
            assert (
                await connection.fetchval(
                    "SELECT delivered_at IS NULL "
                    "FROM public.sim_archive_rehydration_outbox "
                    "WHERE source_topic = $1 AND source_partition = $2 "
                    "AND source_offset = $3",
                    *selected.source_key,
                )
                is True
            )

        assert await outbox.enqueue_once(unrelated) is True
        with pytest.raises(ValueError, match="unrelated archive rows"):
            await outbox.require_only_selected_keys((selected,))
        broker.fail = False
        assert await outbox.relay_selected_once((selected,)) == 1
        assert broker.acked == [
            (
                selected.target_topic,
                selected.key,
                selected.value,
                list(selected.headers),
                selected.timestamp_ms,
            )
        ]
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT source_offset, delivered_at IS NOT NULL AS delivered "
                "FROM public.sim_archive_rehydration_outbox ORDER BY source_offset"
            )
        assert [(row["source_offset"], row["delivered"]) for row in rows] == [
            (11, True),
            (12, False),
        ]
        assert await outbox.relay_selected_once((selected,)) == 0
        assert len(broker.acked) == 1
    finally:
        await pool.close()
