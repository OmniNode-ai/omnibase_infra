# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerPostgresRuntimeManifestInsert (OMN-11197).

Covers:
    - Successful INSERT produces ModelBackendResult(success=True)
    - Duplicate row (ON CONFLICT DO NOTHING, fetchrow returns None) still succeeds
    - Correct SQL parameters passed to fetchrow
    - asyncpg connection pool error returns ModelBackendResult(success=False)

OMN-17296: these drive ``handle(envelope)``, the shape auto-wiring actually
dispatches. They previously called ``handle(payload, correlation_id)`` — a
two-argument signature no dispatch path ever used — so the whole file was green
while every manifest event on the dev lane dead-lettered with
``TypeError: handle() missing 1 required positional argument: 'correlation_id'``.
The fold from event to INSERT payload is now covered end to end here rather than
being assumed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.runtime_manifest.model_manifest_contract import (
    ModelManifestContract,
)
from omnibase_core.models.runtime_manifest.model_manifest_handler import (
    ModelManifestHandler,
)
from omnibase_infra.nodes.node_runtime_manifest_reducer.handlers.handler_postgres_runtime_manifest_insert import (
    SQL_INSERT_RUNTIME_MANIFEST,
    HandlerPostgresRuntimeManifestInsert,
)
from omnibase_infra.nodes.node_runtime_manifest_reducer.models.model_payload_insert_runtime_manifest import (
    ModelPayloadInsertRuntimeManifest,
)
from omnibase_infra.runtime.models.model_runtime_manifest_published import (
    ModelRuntimeManifestPublished,
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


def _make_event(**overrides: object) -> ModelRuntimeManifestPublished:
    """A published boot manifest, the payload the subscribed topic carries."""
    defaults: dict[str, object] = {
        "runtime_profile": "main",
        "contracts": (
            ModelManifestContract(
                name="node_foo",
                version="1.0.0",
                node_type="EFFECT_GENERIC",
                contract_hash="c1",
            ),
        ),
        "owned_command_topics": frozenset({"onex.cmd.platform.register.v1"}),
        "subscribed_event_topics": frozenset({"onex.evt.platform.node-registered.v1"}),
        "handlers": (
            ModelManifestHandler(
                name="HandlerFoo",
                module_path="foo.bar",
                routing_strategy="payload_type_match",
            ),
        ),
        "image_digest": None,
        "started_at": datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return ModelRuntimeManifestPublished(**defaults)  # type: ignore[arg-type]


def _envelope(
    event: ModelRuntimeManifestPublished,
) -> ModelEventEnvelope[ModelRuntimeManifestPublished]:
    return ModelEventEnvelope(
        payload=event,
        correlation_id=uuid4(),
        event_type="omnibase-infra.runtime-manifest-published",
        source_tool="service_kernel",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_success_returns_backend_result_true() -> None:
    """Successful INSERT: returns ModelBackendResult(success=True)."""
    pool = _make_pool(fetchrow_return=_make_record(row_id=42))
    handler = HandlerPostgresRuntimeManifestInsert(pool)

    result = await handler.handle(_envelope(_make_event()))

    assert result.success is True
    assert result.backend_id == "postgres"
    assert not result.error  # None on success


@pytest.mark.asyncio
async def test_duplicate_insert_returns_success() -> None:
    """ON CONFLICT DO NOTHING (fetchrow returns None) still succeeds."""
    pool = _make_pool(fetchrow_return=None)
    handler = HandlerPostgresRuntimeManifestInsert(pool)

    result = await handler.handle(_envelope(_make_event()))

    assert result.success is True


@pytest.mark.asyncio
async def test_correct_sql_called() -> None:
    """Verifies fetchrow is called with the canonical SQL statement."""
    pool = _make_pool(fetchrow_return=_make_record())
    handler = HandlerPostgresRuntimeManifestInsert(pool)

    await handler.handle(_envelope(_make_event()))

    conn = pool._test_conn
    conn.fetchrow.assert_called_once()
    called_sql = conn.fetchrow.call_args[0][0]
    assert called_sql == SQL_INSERT_RUNTIME_MANIFEST


@pytest.mark.asyncio
async def test_correct_positional_args_passed() -> None:
    """Verifies positional args match the EVENT's fields in the column order.

    The three hashes are derived, not supplied: two are the publisher's own
    computed fields and the third is the whole-manifest hash. Asserting them
    against the event's values is what proves the fold copied rather than
    recomputed or dropped them.
    """
    pool = _make_pool(fetchrow_return=_make_record())
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    event = _make_event(runtime_profile="staging", image_digest="sha256:abc")

    await handler.handle(_envelope(event))

    conn = pool._test_conn
    args = conn.fetchrow.call_args[0]
    # args[0] = SQL, args[1..] = positional params
    assert args[1] == "staging"  # runtime_profile
    assert args[2] == event.contract_hash
    assert args[3] == event.topology_hash
    assert args[4] == ModelPayloadInsertRuntimeManifest.manifest_hash_for(event)
    assert args[12] == "sha256:abc"  # image_digest
    assert args[13] == event.started_at  # started_at


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

    result = await handler.handle(_envelope(_make_event()))

    assert result.success is False
    assert result.error != ""


@pytest.mark.asyncio
async def test_topics_are_sorted_in_jsonb() -> None:
    """owned_command_topics and subscribed_event_topics are sorted before serialization.

    The wire type is a frozenset, whose iteration order is not stable across
    processes, so without the sort two rows describing an identical topology
    would differ by column bytes.
    """
    import json

    pool = _make_pool(fetchrow_return=_make_record())
    handler = HandlerPostgresRuntimeManifestInsert(pool)
    event = _make_event(
        owned_command_topics=frozenset({"onex.cmd.z.v1", "onex.cmd.a.v1"}),
        subscribed_event_topics=frozenset({"onex.evt.z.v1", "onex.evt.a.v1"}),
    )
    await handler.handle(_envelope(event))

    conn = pool._test_conn
    args = conn.fetchrow.call_args[0]
    owned = json.loads(args[6])  # owned_command_topics
    subscribed = json.loads(args[7])  # subscribed_event_topics
    assert owned == ["onex.cmd.a.v1", "onex.cmd.z.v1"]
    assert subscribed == ["onex.evt.a.v1", "onex.evt.z.v1"]


@pytest.mark.asyncio
async def test_handler_type_and_category() -> None:
    """handler_type and handler_category expose correct enum values."""
    from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

    pool = _make_pool()
    handler = HandlerPostgresRuntimeManifestInsert(pool)

    assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category == EnumHandlerTypeCategory.EFFECT
