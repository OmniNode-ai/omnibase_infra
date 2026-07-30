# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.idempotency import StoreIdempotencySqlite

pytestmark = pytest.mark.asyncio


async def test_marker_survives_store_restart(tmp_path: Path) -> None:
    path = tmp_path / "gateway-delivery.sqlite3"
    envelope_id = uuid4()
    first = StoreIdempotencySqlite(path)
    await first.start()
    await first.mark_processed(envelope_id, domain="tenant-acme")
    await first.close()

    restarted = StoreIdempotencySqlite(path)
    await restarted.start()

    assert await restarted.is_processed(envelope_id, domain="tenant-acme") is True
    assert await restarted.is_processed(envelope_id, domain="tenant-other") is False


async def test_check_and_record_is_atomic_for_concurrent_callers(
    tmp_path: Path,
) -> None:
    store = StoreIdempotencySqlite(tmp_path / "gateway-delivery.sqlite3")
    await store.start()
    envelope_id = uuid4()

    import asyncio

    results = await asyncio.gather(
        *(store.check_and_record(envelope_id, domain="tenant-acme") for _ in range(8))
    )

    assert results.count(True) == 1
    assert results.count(False) == 7


async def test_cleanup_uses_durable_processed_timestamp(tmp_path: Path) -> None:
    store = StoreIdempotencySqlite(tmp_path / "gateway-delivery.sqlite3")
    await store.start()
    expired = uuid4()
    current = uuid4()
    await store.mark_processed(
        expired,
        domain="tenant-acme",
        processed_at=datetime.now(UTC) - timedelta(hours=25),
    )
    await store.mark_processed(current, domain="tenant-acme")

    removed = await store.cleanup_expired(ttl_seconds=24 * 60 * 60)

    assert removed == 1
    assert await store.is_processed(expired, domain="tenant-acme") is False
    assert await store.is_processed(current, domain="tenant-acme") is True


async def test_operations_fail_closed_before_store_start(tmp_path: Path) -> None:
    store = StoreIdempotencySqlite(tmp_path / "gateway-delivery.sqlite3")

    with pytest.raises(RuntimeError, match="not started"):
        await store.is_processed(uuid4(), domain="tenant-acme")
