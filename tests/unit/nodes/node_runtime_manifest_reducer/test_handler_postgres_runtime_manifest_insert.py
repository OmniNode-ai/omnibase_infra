# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerPostgresRuntimeManifestInsert (OMN-11197).

Covers:
    - Successful INSERT produces ModelBackendResult(success=True)
    - Duplicate row (ON CONFLICT DO NOTHING, fetchrow returns None) still succeeds
    - Correct SQL parameters passed to fetchrow
    - asyncpg connection pool error returns ModelBackendResult(success=False)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.nodes.node_runtime_manifest_reducer.handlers.handler_postgres_runtime_manifest_insert import (
    SQL_INSERT_RUNTIME_MANIFEST,
    HandlerPostgresRuntimeManifestInsert,
)
from omnibase_infra.nodes.node_runtime_manifest_reducer.models.model_payload_insert_runtime_manifest import (
    ModelPayloadInsertRuntimeManifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(fetchrow_return: object = None) -> MagicMock:
    """Build a mock asyncpg.Pool.

    fetchrow_return: value returned by conn.fetchrow (None = ON CONFLICT DO NOTHING).
    """
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    pool._test_conn = conn
    return pool


def _make_record(row_id: int = 1) -> MagicMock:
    rec = MagicMock()
    rec.__getitem__ = MagicMock(side_effect=lambda k: row_id if k == "id" else None)
    return rec


def _make_payload(**overrides: object) -> ModelPayloadInsertRuntimeManifest:
    defaults: dict[str, object] = {
        "runtime_profile": "main",
        "contract_hash": "abc123",
        "topology_hash": "topo456",
        "manifest_hash": "mfst789",
        "contracts": [
            {
                "name": "node_foo",
                "version": "1.0.0",
                "node_type": "EFFECT_GENERIC",
                "contract_hash": "c1",
            }
        ],
        "owned_command_topics": ["onex.cmd.platform.register.v1"],
        "subscribed_event_topics": ["onex.evt.platform.node-registered.v1"],
        "handlers": [
            {
                "name": "HandlerFoo",
                "module_path": "foo.bar",
                "routing_strategy": "payload_type_match",
            }
        ],
        "skipped_contracts": [],
        "failed_contracts": [],
        "ownership_violations": [],
        "image_digest": None,
        "started_at": datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return ModelPayloadInsertRuntimeManifest(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_success_returns_backend_result_true() -> None:
    """Successful INSERT: returns ModelBackendResult(success=True)."""
    pool = _make_pool(fetchrow_return=_make_record(row_id=42))
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    payload = _make_payload()

    result = await handler.handle(payload, uuid4())

    assert result.success is True
    assert result.backend_id == "postgres"
    assert not result.error  # None on success


@pytest.mark.asyncio
async def test_duplicate_insert_returns_success() -> None:
    """ON CONFLICT DO NOTHING (fetchrow returns None) still succeeds."""
    pool = _make_pool(fetchrow_return=None)
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    payload = _make_payload()

    result = await handler.handle(payload, uuid4())

    assert result.success is True


@pytest.mark.asyncio
async def test_correct_sql_called() -> None:
    """Verifies fetchrow is called with the canonical SQL statement."""
    pool = _make_pool(fetchrow_return=_make_record())
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    payload = _make_payload()
    correlation_id = uuid4()

    await handler.handle(payload, correlation_id)

    conn = pool._test_conn
    conn.fetchrow.assert_called_once()
    called_sql = conn.fetchrow.call_args[0][0]
    assert called_sql == SQL_INSERT_RUNTIME_MANIFEST


@pytest.mark.asyncio
async def test_correct_positional_args_passed() -> None:
    """Verifies positional args match payload fields in correct order."""
    pool = _make_pool(fetchrow_return=_make_record())
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    payload = _make_payload(
        runtime_profile="staging",
        contract_hash="ch",
        topology_hash="th",
        manifest_hash="mh",
        image_digest="sha256:abc",
    )
    await handler.handle(payload, uuid4())

    conn = pool._test_conn
    args = conn.fetchrow.call_args[0]
    # args[0] = SQL, args[1..] = positional params
    assert args[1] == "staging"  # runtime_profile
    assert args[2] == "ch"  # contract_hash
    assert args[3] == "th"  # topology_hash
    assert args[4] == "mh"  # manifest_hash
    assert args[12] == "sha256:abc"  # image_digest
    assert args[13] == payload.started_at  # started_at


@pytest.mark.asyncio
async def test_pool_error_returns_backend_result_false() -> None:
    """asyncpg connection error produces ModelBackendResult(success=False)."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=Exception("connection refused"))

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)

    handler = HandlerPostgresRuntimeManifestInsert(pool)
    payload = _make_payload()

    result = await handler.handle(payload, uuid4())

    assert result.success is False
    assert result.error != ""


@pytest.mark.asyncio
async def test_topics_are_sorted_in_jsonb() -> None:
    """owned_command_topics and subscribed_event_topics are sorted before serialization."""
    import json

    pool = _make_pool(fetchrow_return=_make_record())
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    payload = _make_payload(
        owned_command_topics=["onex.cmd.z.v1", "onex.cmd.a.v1"],
        subscribed_event_topics=["onex.evt.z.v1", "onex.evt.a.v1"],
    )
    await handler.handle(payload, uuid4())

    conn = pool._test_conn
    args = conn.fetchrow.call_args[0]
    owned = json.loads(args[6])  # owned_command_topics
    subscribed = json.loads(args[7])  # subscribed_event_topics
    assert owned == sorted(owned)
    assert subscribed == sorted(subscribed)


@pytest.mark.asyncio
async def test_handler_type_and_category() -> None:
    """handler_type and handler_category expose correct enum values."""
    from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

    pool = _make_pool()
    handler = HandlerPostgresRuntimeManifestInsert(pool)

    assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category == EnumHandlerTypeCategory.EFFECT
