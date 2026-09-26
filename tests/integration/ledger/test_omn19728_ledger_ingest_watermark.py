# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A committed ingest cursor survives late Kafka offsets and duplicate delivery."""

from __future__ import annotations

import asyncio
import getpass
import shutil
import socket
import subprocess
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_MIGRATION = REPO_ROOT / "docker/migrations/forward/044_create_event_ledger.sql"
CURSOR_MIGRATION = (
    REPO_ROOT / "docker/migrations/forward/108_add_event_ledger_ingest_watermark.sql"
)
TOPIC = "onex.evt.test.ingest-watermark.v1"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
async def local_postgres(
    tmp_path: Path,
) -> AsyncGenerator[tuple[asyncpg.Connection, dict[str, str | int]], None]:
    """Run these writes in an isolated local cluster, never the lab database."""
    import asyncpg

    initdb = shutil.which("initdb")
    pg_ctl = shutil.which("pg_ctl")
    psql = shutil.which("psql")
    if initdb is None or pg_ctl is None or psql is None:
        pytest.skip("Local PostgreSQL binaries are unavailable")

    data_dir = tmp_path / "pgdata"
    socket_dir = Path(tempfile.mkdtemp(prefix="omn19728-", dir="/tmp"))
    port = _free_port()
    subprocess.run(
        [initdb, "-D", str(data_dir), "--auth=trust", "--no-instructions"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            pg_ctl,
            "-D",
            str(data_dir),
            "-l",
            str(tmp_path / "postgres.log"),
            "-o",
            f"-h '' -k {socket_dir} -p {port}",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    connection_args: dict[str, str | int] = {
        "database": "postgres",
        "user": getpass.getuser(),
        "host": str(socket_dir),
        "port": port,
    }
    try:
        subprocess.run(
            [
                psql,
                "-X",
                "-v",
                "ON_ERROR_STOP=1",
                "-h",
                str(socket_dir),
                "-p",
                str(port),
                "-d",
                "postgres",
                "-f",
                str(BASE_MIGRATION),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        connection = await asyncpg.connect(**connection_args)
        try:
            yield connection, connection_args
        finally:
            await connection.close()
    finally:
        subprocess.run(
            [pg_ctl, "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=True,
            capture_output=True,
            text=True,
        )
        socket_dir.rmdir()


async def _append(
    connection: asyncpg.Connection,
    *,
    offset: int,
    body: bytes | None = None,
) -> asyncpg.Record:
    return await connection.fetchrow(
        """
        SELECT ledger_entry_id, duplicate, ingest_epoch, ingest_seq
        FROM public.append_event_ledger_with_watermark(
            $1::text, $2::integer, $3::bigint, $4::bytea, $5::bytea,
            $6::jsonb, $7::uuid, $8::uuid, $9::text, $10::text, $11::timestamptz
        )
        """,
        TOPIC,
        0,
        offset,
        None,
        body if body is not None else f"event-{offset}".encode(),
        "{}",
        None,
        None,
        None,
        None,
        None,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_cursor_is_commit_ordered_and_legacy_is_unversioned(
    local_postgres: tuple[asyncpg.Connection, dict[str, str | int]],
) -> None:
    """A later lower Kafka offset stays outside an earlier ingest bound."""
    import asyncpg

    connection, _ = local_postgres
    await connection.execute(
        "INSERT INTO public.event_ledger (topic, partition, kafka_offset, event_value) "
        "VALUES ($1, 0, 10, $2)",
        TOPIC,
        b"legacy-10",
    )
    await connection.execute(CURSOR_MIGRATION.read_text(encoding="utf-8"))
    baseline = await connection.fetchrow(
        "SELECT legacy_max_kafka_offset, legacy_row_count "
        "FROM public.ledger_ingest_legacy_baseline "
        "WHERE topic=$1 AND partition=0",
        TOPIC,
    )
    assert baseline is not None
    assert tuple(baseline) == (10, 1)

    high = await _append(connection, offset=12)
    assert high is not None
    assert (high["duplicate"], high["ingest_epoch"], high["ingest_seq"]) == (
        False,
        1,
        1,
    )
    bounded_before = await connection.fetch(
        "SELECT kafka_offset FROM public.event_ledger "
        "WHERE topic=$1 AND partition=0 AND ingest_epoch=1 AND ingest_seq<=1 "
        "ORDER BY ingest_seq",
        TOPIC,
    )
    assert [row["kafka_offset"] for row in bounded_before] == [12]

    late_low = await _append(connection, offset=11)
    assert late_low is not None
    assert late_low["ingest_seq"] == 2
    bounded_after = await connection.fetch(
        "SELECT kafka_offset FROM public.event_ledger "
        "WHERE topic=$1 AND partition=0 AND ingest_epoch=1 AND ingest_seq<=1 "
        "ORDER BY ingest_seq",
        TOPIC,
    )
    assert bounded_after == bounded_before

    duplicate = await _append(connection, offset=12)
    assert duplicate is not None
    assert (duplicate["duplicate"], duplicate["ingest_seq"]) == (True, 1)
    with pytest.raises(asyncpg.PostgresError):
        await _append(connection, offset=12, body=b"conflicting-position")

    legacy = await connection.fetchrow(
        "SELECT ingest_epoch, ingest_seq FROM public.event_ledger "
        "WHERE topic=$1 AND partition=0 AND kafka_offset=10",
        TOPIC,
    )
    assert legacy is not None
    assert (legacy["ingest_epoch"], legacy["ingest_seq"]) == (None, None)
    legacy_duplicate = await _append(connection, offset=10, body=b"legacy-10")
    assert legacy_duplicate is not None
    assert (
        legacy_duplicate["duplicate"],
        legacy_duplicate["ingest_epoch"],
        legacy_duplicate["ingest_seq"],
    ) == (True, None, None)
    with pytest.raises(asyncpg.PostgresError):
        await connection.execute(
            "UPDATE public.event_ledger SET ingest_epoch=1 "
            "WHERE topic=$1 AND partition=0 AND kafka_offset=10",
            TOPIC,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_sequence_serializes_concurrent_commits(
    local_postgres: tuple[asyncpg.Connection, dict[str, str | int]],
) -> None:
    """The second writer waits for the first transaction's partition lock."""
    import asyncpg

    connection, connection_args = local_postgres
    await connection.execute(CURSOR_MIGRATION.read_text(encoding="utf-8"))
    first = await asyncpg.connect(**connection_args)
    second = await asyncpg.connect(**connection_args)
    try:
        transaction = first.transaction()
        await transaction.start()
        first_row = await _append(first, offset=20)
        assert first_row is not None
        pending = asyncio.create_task(_append(second, offset=19))
        await asyncio.sleep(0.1)
        assert not pending.done()
        await transaction.commit()
        second_row = await asyncio.wait_for(pending, timeout=5)
        assert second_row is not None
        assert first_row["ingest_seq"] < second_row["ingest_seq"]
    finally:
        await first.close()
        await second.close()
