# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression guard: asyncpg UUID rows must survive cast to stdlib UUID.

Context:
    OMN-9041 — ``UUID(row["node_id"])`` crashed in production because asyncpg
    decoded the VARCHAR ``node_id`` column as an ``asyncpg.pgproto.pgproto.UUID``
    object (not a ``str``). stdlib ``uuid.UUID.__init__`` calls ``.replace()``
    on its first positional arg, which the asyncpg UUID type does not
    implement, so the cast raised ``AttributeError``.

    Fix: wrap the row value with ``str()`` before passing to stdlib ``UUID``.

    This test exercises ``HandlerRegistrationStoragePostgres.query_registrations``
    end-to-end against an in-process stub pool whose rows contain an asyncpg
    pgproto UUID. Pre-fix this test fails with ``AttributeError`` inside the
    row conversion loop; post-fix it passes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelStorageQuery,
)


class _FakeAsyncpgUUID:
    """Stand-in for ``asyncpg.pgproto.pgproto.UUID``.

    Mimics the critical failure-mode invariant: has no ``.replace()``
    attribute, so stdlib ``uuid.UUID(fake)`` raises ``AttributeError`` unless
    the caller coerces to ``str`` first. ``__str__`` returns a canonical
    hex UUID, matching asyncpg's real behavior.
    """

    def __init__(self, canonical: str) -> None:
        self._canonical = canonical

    def __str__(self) -> str:
        return self._canonical


class _StubRecord(dict[str, Any]):
    """Minimal ``asyncpg.Record`` stand-in supporting ``row["col"]`` access."""


class _StubConnection:
    def __init__(self, rows: list[_StubRecord], count: int) -> None:
        self._rows = rows
        self._count = count

    async def fetch(self, _query: str, *_params: Any) -> list[_StubRecord]:
        return self._rows

    async def fetchval(self, _query: str, *_params: Any) -> int:
        return self._count


class _StubAcquireCtx:
    def __init__(self, conn: _StubConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _StubConnection:
        return self._conn

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _StubPool:
    def __init__(self, conn: _StubConnection) -> None:
        self._conn = conn

    def acquire(self) -> _StubAcquireCtx:
        return _StubAcquireCtx(self._conn)


@pytest.mark.asyncio
async def test_query_registrations_handles_asyncpg_pgproto_uuid() -> None:
    """query_registrations must survive asyncpg returning pgproto UUID objects.

    Regression guard for OMN-9041. Without the ``str()`` coercion in the
    ModelRegistrationRecord construction, this test raises AttributeError
    inside the row conversion loop.
    """
    canonical = "01010101-0101-0101-0101-010101010101"
    fake_node_id = _FakeAsyncpgUUID(canonical)

    row = _StubRecord(
        node_id=fake_node_id,
        node_type="effect",
        node_version="1.0.0",
        capabilities="[]",
        endpoints="{}",
        metadata="{}",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    handler = HandlerRegistrationStoragePostgres(
        container=MagicMock(spec=ModelONEXContainer),
        host="unused",
        port=5432,
        database="unused",
        user="unused",
        password="unused",
    )

    stub_pool = _StubPool(_StubConnection(rows=[row], count=1))

    async def _fake_ensure_pool(correlation_id: UUID | None = None) -> _StubPool:
        return stub_pool

    handler._ensure_pool = _fake_ensure_pool  # type: ignore[method-assign]

    result = await handler.query_registrations(
        query=ModelStorageQuery(),
        correlation_id=uuid4(),
    )

    assert result.success, f"Query should succeed; got error: {result.error}"
    assert len(result.records) == 1
    record = result.records[0]
    assert isinstance(record.node_id, UUID), (
        "node_id must be stdlib uuid.UUID after cast"
    )
    assert str(record.node_id) == canonical
