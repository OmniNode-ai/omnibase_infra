# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for overseer scoped wrapper adapters.

Tests the four required cases from OMN-8065:
    - test_ticket_create_succeeds_when_allowed
    - test_ticket_create_raises_when_denied
    - test_event_bus_publish_succeeds_when_allowed
    - test_llm_chat_completion_raises_when_denied
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.adapters.overseer.adapter_event_bus_scoped import (
    AdapterEventBusScoped,
)
from omnibase_infra.adapters.overseer.adapter_llm_provider_scoped import (
    AdapterLlmProviderScoped,
)
from omnibase_infra.adapters.overseer.adapter_ticket_service_linear_scoped import (
    AdapterTicketLinearScoped,
)
from omnibase_infra.errors import InvariantViolation
from omnibase_infra.protocols import ProtocolEventBusLike

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ticket_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.create_ticket = AsyncMock(return_value="OMN-9999")
    adapter.get_ticket = AsyncMock(return_value={"id": "OMN-9999", "title": "test"})
    adapter.list_tickets = AsyncMock(return_value=[])
    adapter.get_ticket_status = AsyncMock(return_value="In Progress")
    adapter.update_ticket_status = AsyncMock(return_value=True)
    adapter.add_comment = AsyncMock(return_value="comment-123")
    adapter.health_check = AsyncMock(return_value=True)
    adapter.close = AsyncMock(return_value=None)
    return adapter


@pytest.fixture
def mock_event_bus() -> MagicMock:
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.publish = AsyncMock(return_value=None)
    bus.publish_envelope = AsyncMock(return_value=None)
    bus.subscribe = AsyncMock(return_value=AsyncMock())
    bus.close = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def mock_llm_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.provider_name = "mock-provider"
    adapter.provider_type = "local"
    adapter.is_available = True
    adapter.generate_async = AsyncMock(return_value=MagicMock())
    adapter.generate = AsyncMock(return_value=MagicMock())
    adapter.health_check = AsyncMock(return_value=MagicMock(is_healthy=True))
    return adapter


# ---------------------------------------------------------------------------
# Ticket service — allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ticket_create_succeeds_when_allowed(
    mock_ticket_adapter: MagicMock,
) -> None:
    scoped = AdapterTicketLinearScoped(
        inner=mock_ticket_adapter,
        allowed_actions=frozenset({"create_ticket"}),
    )

    result = await scoped.create_ticket(
        title="New ticket",
        description="Test ticket creation",
    )

    assert result == "OMN-9999"
    mock_ticket_adapter.create_ticket.assert_awaited_once()


# ---------------------------------------------------------------------------
# Ticket service — denied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ticket_create_raises_when_denied(
    mock_ticket_adapter: MagicMock,
) -> None:
    scoped = AdapterTicketLinearScoped(
        inner=mock_ticket_adapter,
        allowed_actions=frozenset({"get_ticket", "list_tickets"}),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        await scoped.create_ticket(
            title="New ticket",
            description="Denied by scope",
        )

    error = exc_info.value
    assert error.action_name == "create_ticket"
    assert error.protocol_domain == "ticket_service"
    mock_ticket_adapter.create_ticket.assert_not_called()


# ---------------------------------------------------------------------------
# Event bus — publish allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_bus_publish_succeeds_when_allowed(
    mock_event_bus: MagicMock,
) -> None:
    scoped = AdapterEventBusScoped(
        inner=mock_event_bus,
        allowed_actions=frozenset({"publish", "publish_envelope"}),
    )

    await scoped.publish(
        topic="onex.evt.test.v1",
        key=b"key",
        value=b'{"event": "test"}',
    )

    mock_event_bus.publish.assert_awaited_once_with(
        topic="onex.evt.test.v1",
        key=b"key",
        value=b'{"event": "test"}',
    )


@pytest.mark.asyncio
async def test_event_bus_publish_raises_when_denied(
    mock_event_bus: MagicMock,
) -> None:
    scoped = AdapterEventBusScoped(
        inner=mock_event_bus,
        allowed_actions=frozenset({"subscribe"}),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        await scoped.publish(
            topic="onex.evt.test.v1",
            key=None,
            value=b"{}",
        )

    error = exc_info.value
    assert error.action_name == "publish"
    assert error.protocol_domain == "event_bus"
    mock_event_bus.publish.assert_not_called()


# ---------------------------------------------------------------------------
# LLM provider — generate_async denied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_chat_completion_raises_when_denied(
    mock_llm_adapter: MagicMock,
) -> None:
    scoped = AdapterLlmProviderScoped(
        inner=mock_llm_adapter,
        allowed_actions=frozenset({"health_check", "get_available_models"}),
    )

    request = MagicMock()

    with pytest.raises(InvariantViolation) as exc_info:
        await scoped.generate_async(request)

    error = exc_info.value
    assert error.action_name == "generate_async"
    assert error.protocol_domain == "llm_provider"
    mock_llm_adapter.generate_async.assert_not_called()


@pytest.mark.asyncio
async def test_llm_chat_completion_succeeds_when_allowed(
    mock_llm_adapter: MagicMock,
) -> None:
    scoped = AdapterLlmProviderScoped(
        inner=mock_llm_adapter,
        allowed_actions=frozenset({"generate_async", "health_check"}),
    )

    request = MagicMock()
    await scoped.generate_async(request)

    mock_llm_adapter.generate_async.assert_awaited_once_with(request)


# ---------------------------------------------------------------------------
# InvariantViolation structure verification
# ---------------------------------------------------------------------------


def test_invariant_violation_carries_allowed_actions(
    mock_ticket_adapter: MagicMock,
) -> None:
    allowed = frozenset({"get_ticket", "list_tickets"})
    scoped = AdapterTicketLinearScoped(
        inner=mock_ticket_adapter,
        allowed_actions=allowed,
    )

    try:
        scoped._check("create_ticket")
        pytest.fail("Expected InvariantViolation")
    except InvariantViolation as exc:
        assert exc.action_name == "create_ticket"
        from typing import cast

        raw = cast("tuple[str, ...]", exc.model.context["allowed_actions"])
        assert set(raw) == set(allowed)


def test_scoped_adapter_passthrough_properties(mock_llm_adapter: MagicMock) -> None:
    scoped = AdapterLlmProviderScoped(
        inner=mock_llm_adapter,
        allowed_actions=frozenset(),
    )

    assert scoped.provider_name == "mock-provider"
    assert scoped.provider_type == "local"
    assert scoped.is_available is True
