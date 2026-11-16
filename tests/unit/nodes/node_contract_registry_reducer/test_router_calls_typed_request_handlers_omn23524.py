# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Router calls the canonical typed-request handler shape (OMN-23524).

OMN-21697 moved the six persistence-effect handlers to handle(request), but the
router still called handle(payload, correlation_id), so every intent failed
with a TypeError logged as "Error executing intent". AsyncMock handlers hid it.
These tests pin both sides of the seam with real objects.
"""

from __future__ import annotations

import inspect
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.nodes.node_contract_persistence_effect import (
    handlers as persistence_handlers,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers import (
    HandlerPostgresMarkStale,
)
from omnibase_infra.nodes.node_contract_registry_reducer.contract_registration_event_router import (
    ContractRegistrationEventRouter,
    ProtocolIntentEffect,
)
from omnibase_infra.nodes.node_contract_registry_reducer.models.model_payload_mark_stale import (
    ModelPayloadMarkStale,
)


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
async def test_mark_stale_intent_reaches_the_real_handler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A mark_stale intent dispatches into the real handler with no errors."""
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


def test_every_persistence_handler_takes_one_request() -> None:
    """Every exported handler class (and the protocol) is handle(self, request)."""
    handler_classes = tuple(
        getattr(persistence_handlers, name)
        for name in persistence_handlers.__all__
        if inspect.isclass(getattr(persistence_handlers, name))
    )
    assert len(handler_classes) == 6
    for handler_class in handler_classes:
        assert list(inspect.signature(handler_class.handle).parameters) == [
            "self",
            "request",
        ], handler_class.__name__
    assert list(inspect.signature(ProtocolIntentEffect.handle).parameters) == [
        "self",
        "request",
    ]
