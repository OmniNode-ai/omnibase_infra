# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23524: the router dispatches every routed intent into the real handler.

The persistence-effect handlers take one typed request (OMN-21697), so the
router must call handle(payload) — not handle(payload, correlation_id). On the
old call shape the real handler raised a TypeError that the router logged as
"Error executing intent", and every intent silently did nothing. Each test here
dispatches one intent through ContractRegistrationEventRouter._execute_intents
into the real handler over a fake asyncpg-like pool and requires the SQL path
to run (conn.execute awaited) with no "Error executing intent" warning.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.nodes.node_contract_persistence_effect.handlers import (
    HandlerPostgresHeartbeat,
    HandlerPostgresMarkStale,
)
from omnibase_infra.nodes.node_contract_registry_reducer.contract_registration_event_router import (
    ContractRegistrationEventRouter,
)
from omnibase_infra.nodes.node_contract_registry_reducer.models.model_payload_mark_stale import (
    ModelPayloadMarkStale,
)
from omnibase_infra.nodes.node_contract_registry_reducer.models.model_payload_update_heartbeat import (
    ModelPayloadUpdateHeartbeat,
)

pytestmark = pytest.mark.integration


class _FakePoolAcquire:
    """Async context manager whose __aenter__ yields the fake connection."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _make_router(effect_handlers: dict[str, Any]) -> ContractRegistrationEventRouter:
    """Build a router with mocks for the parts these tests do not exercise."""
    return ContractRegistrationEventRouter(
        container=MagicMock(),
        reducer=MagicMock(),
        effect_handlers=effect_handlers,
    )


@pytest.mark.asyncio
async def test_mark_stale_intent_through_the_router_runs_the_real_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A mark_stale intent dispatches into the real handler with one argument."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="UPDATE 2")
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_FakePoolAcquire(conn))
    router = _make_router({"postgres.mark_stale": HandlerPostgresMarkStale(pool)})

    payload = ModelPayloadMarkStale(
        correlation_id=uuid4(),
        stale_cutoff=datetime(2026, 1, 15, 11, 0, 0, tzinfo=UTC),
        checked_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
    )
    intent = ModelIntent(
        intent_type="extension",
        target="postgres://contracts/stale",
        payload=payload,
    )

    with caplog.at_level(logging.WARNING):
        await router._execute_intents((intent,), uuid4())

    conn.execute.assert_awaited_once()
    assert [
        record
        for record in caplog.records
        if "Error executing intent" in record.getMessage()
    ] == []


@pytest.mark.asyncio
async def test_heartbeat_intent_through_the_router_runs_the_real_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An update_heartbeat intent dispatches into the real handler."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="UPDATE 1")
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_FakePoolAcquire(conn))
    router = _make_router({"postgres.update_heartbeat": HandlerPostgresHeartbeat(pool)})

    payload = ModelPayloadUpdateHeartbeat(
        correlation_id=uuid4(),
        contract_id="registry-api:1.2.3",
        node_name="registry-api",
        last_seen_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
    )
    intent = ModelIntent(
        intent_type="extension",
        target="postgres://contracts/registry-api:1.2.3/heartbeat",
        payload=payload,
    )

    with caplog.at_level(logging.WARNING):
        await router._execute_intents((intent,), uuid4())

    conn.execute.assert_awaited_once()
    assert [
        record
        for record in caplog.records
        if "Error executing intent" in record.getMessage()
    ] == []
