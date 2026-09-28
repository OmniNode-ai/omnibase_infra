# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19803: real-Postgres commit-order ledger watermark proof."""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse
from uuid import uuid4

import asyncpg
import pytest


def _isolated_dsn() -> str:
    dsn = os.environ.get("OMN19803_TEST_DSN")
    if dsn is None:
        pytest.skip("requires a dedicated disposable OMN-19803 PostgreSQL DSN")
    parsed = urlparse(dsn)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.hostname != "127.0.0.1"
        or parsed.username != "postgres"
        or parsed.password is None
        or parsed.path != "/postgres"
        or parsed.port is None
    ):
        raise ValueError("OMN19803_TEST_DSN must target dedicated loopback postgres")
    return dsn


async def _connect_disposable(dsn: str) -> asyncpg.Connection:
    connection = await asyncpg.connect(dsn)
    marker = await connection.fetchval(
        "SELECT to_regclass('omn19803_test_guard.disposable_instance') IS NOT NULL"
    )
    if marker is not True:
        await connection.close()
        raise ValueError("OMN19803_TEST_DSN lacks disposable DB marker")
    return connection


@pytest.mark.integration
@pytest.mark.asyncio
async def test_watermark_tracks_commit_order_not_kafka_offset() -> None:
    connection = await _connect_disposable(_isolated_dsn())
    topic = f"omn19803.test.{uuid4()}"
    try:
        first = await connection.fetchrow(
            "SELECT * FROM public.append_event_ledger_with_watermark("
            "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            topic,
            0,
            10,
            None,
            b"first",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        late_lower = await connection.fetchrow(
            "SELECT * FROM public.append_event_ledger_with_watermark("
            "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            topic,
            0,
            9,
            None,
            b"late",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        duplicate = await connection.fetchrow(
            "SELECT * FROM public.append_event_ledger_with_watermark("
            "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            topic,
            0,
            10,
            None,
            b"first",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        other_partition = await connection.fetchrow(
            "SELECT * FROM public.append_event_ledger_with_watermark("
            "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            topic,
            1,
            1,
            None,
            b"other",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        assert first is not None and late_lower is not None
        assert duplicate is not None and other_partition is not None
        assert (first["ingest_watermark"], late_lower["ingest_watermark"]) == (1, 2)
        assert first["duplicate"] is False
        assert late_lower["duplicate"] is False
        assert duplicate["duplicate"] is True
        assert duplicate["ledger_entry_id"] == first["ledger_entry_id"]
        assert duplicate["ingest_watermark"] == first["ingest_watermark"]
        assert other_partition["ingest_watermark"] == 1
        bounded = await connection.fetch(
            "SELECT kafka_offset FROM public.event_ledger "
            "WHERE topic = $1 AND partition = 0 AND ingest_watermark <= $2 "
            "ORDER BY ingest_watermark",
            topic,
            first["ingest_watermark"],
        )
        assert [row["kafka_offset"] for row in bounded] == [10]
    finally:
        await connection.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_writers_serialize_and_rollback_does_not_advance() -> None:
    dsn = _isolated_dsn()
    first_connection = await _connect_disposable(dsn)
    second_connection = await _connect_disposable(dsn)
    topic = f"omn19803.concurrent.{uuid4()}"

    async def append(connection: asyncpg.Connection, offset: int) -> asyncpg.Record:
        row = await connection.fetchrow(
            "SELECT * FROM public.append_event_ledger_with_watermark("
            "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            topic,
            0,
            offset,
            None,
            b"record",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        assert row is not None
        return row

    try:
        async with first_connection.transaction():
            first = await append(first_connection, 20)
            waiting = asyncio.create_task(append(second_connection, 19))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(waiting), timeout=0.1)
        second = await waiting
        assert first["ingest_watermark"] == 1
        assert second["ingest_watermark"] == 2

        with pytest.raises(RuntimeError, match="roll back test write"):
            async with first_connection.transaction():
                rolled_back = await append(first_connection, 18)
                assert rolled_back["ingest_watermark"] == 3
                raise RuntimeError("roll back test write")
        after_rollback = await append(second_connection, 17)
        assert after_rollback["ingest_watermark"] == 3
    finally:
        await first_connection.close()
        await second_connection.close()
