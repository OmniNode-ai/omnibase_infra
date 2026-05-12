# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for DispatcherDelegationRoutingRequest.

Verifies that the dispatcher correctly routes delegation-routing-request
commands to handler_delegation_routing.delta() and returns typed results.

Related:
    - OMN-10868: Wire delegation routing dispatcher in DI kernel
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumDispatchStatus, EnumMessageCategory
from omnibase_infra.nodes.node_delegation_routing_reducer.dispatchers.dispatcher_delegation_routing_request import (
    DispatcherDelegationRoutingRequest,
)


@pytest.fixture
def dispatcher() -> DispatcherDelegationRoutingRequest:
    return DispatcherDelegationRoutingRequest()


@pytest.fixture
def mock_routing_decision() -> object:
    from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
        ModelRoutingDecision,
    )

    return ModelRoutingDecision(
        correlation_id=uuid4(),
        task_type="test",
        selected_model="test-model",
        selected_backend_id=uuid4(),
        endpoint_url="http://localhost:8000",
        cost_tier="local",
        max_context_tokens=4096,
        system_prompt="You are a test assistant.",
        rationale="Routed to local tier.",
    )


@pytest.mark.unit
class TestDispatcherDelegationRoutingRequest:
    """DispatcherDelegationRoutingRequest unit tests."""

    def test_dispatcher_id(
        self, dispatcher: DispatcherDelegationRoutingRequest
    ) -> None:
        assert dispatcher.dispatcher_id == "dispatcher.delegation.routing-request"

    def test_category_is_command(
        self, dispatcher: DispatcherDelegationRoutingRequest
    ) -> None:
        assert dispatcher.category == EnumMessageCategory.COMMAND

    def test_message_types(
        self, dispatcher: DispatcherDelegationRoutingRequest
    ) -> None:
        assert "omnibase-infra.delegation-routing-request" in dispatcher.message_types
        assert "ModelDelegationRequest" in dispatcher.message_types

    @pytest.mark.asyncio
    async def test_handle_dict_envelope_success(
        self,
        dispatcher: DispatcherDelegationRoutingRequest,
        mock_routing_decision: MagicMock,
    ) -> None:
        from datetime import UTC, datetime

        correlation_id = uuid4()
        envelope = {
            "correlation_id": str(correlation_id),
            "payload": {
                "correlation_id": str(correlation_id),
                "task_type": "test",
                "prompt": "write tests for foo",
                "emitted_at": datetime.now(UTC).isoformat(),
            },
        }

        with patch(
            "omnibase_infra.nodes.node_delegation_routing_reducer.dispatchers.dispatcher_delegation_routing_request.delta",
            return_value=mock_routing_decision,
        ):
            result = await dispatcher.handle(envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert result.dispatcher_id == "dispatcher.delegation.routing-request"
        assert len(result.output_events) == 1
        assert result.output_events[0] == mock_routing_decision

    @pytest.mark.asyncio
    async def test_handle_invalid_payload_returns_invalid_message(
        self,
        dispatcher: DispatcherDelegationRoutingRequest,
    ) -> None:
        envelope = {
            "correlation_id": str(uuid4()),
            "payload": "not-a-dict",
        }

        result = await dispatcher.handle(envelope)

        assert result.status == EnumDispatchStatus.INVALID_MESSAGE
        assert result.dispatcher_id == "dispatcher.delegation.routing-request"

    @pytest.mark.asyncio
    async def test_handle_delta_exception_returns_handler_error(
        self,
        dispatcher: DispatcherDelegationRoutingRequest,
    ) -> None:
        from datetime import UTC, datetime

        correlation_id = uuid4()
        envelope = {
            "correlation_id": str(correlation_id),
            "payload": {
                "correlation_id": str(correlation_id),
                "task_type": "test",
                "prompt": "write tests",
                "emitted_at": datetime.now(UTC).isoformat(),
            },
        }

        with patch(
            "omnibase_infra.nodes.node_delegation_routing_reducer.dispatchers.dispatcher_delegation_routing_request.delta",
            side_effect=RuntimeError("routing failed"),
        ):
            result = await dispatcher.handle(envelope)

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None


@pytest.mark.unit
class TestWireDelegationRoutingDispatcher:
    """wire_delegation_routing_dispatcher must register the routing reducer dispatcher."""

    @pytest.mark.asyncio
    async def test_registers_dispatcher_and_route(self) -> None:
        from unittest.mock import MagicMock

        from omnibase_infra.nodes.node_delegation_orchestrator.wiring import (
            ROUTE_ID_DELEGATION_ROUTING_REQUEST,
            wire_delegation_routing_dispatcher,
        )

        engine = MagicMock()
        engine.register_dispatcher = MagicMock()
        engine.register_route = MagicMock()

        result = await wire_delegation_routing_dispatcher(engine)

        assert engine.register_dispatcher.call_count == 1
        assert engine.register_route.call_count == 1
        assert result["status"] == "success"
        assert "dispatcher.delegation.routing-request" in result["dispatchers"]
        assert ROUTE_ID_DELEGATION_ROUTING_REQUEST in result["routes"]

    @pytest.mark.asyncio
    async def test_registered_dispatcher_handles_correct_message_types(self) -> None:
        from unittest.mock import MagicMock

        from omnibase_infra.nodes.node_delegation_orchestrator.wiring import (
            wire_delegation_routing_dispatcher,
        )

        engine = MagicMock()
        engine.register_dispatcher = MagicMock()
        engine.register_route = MagicMock()

        await wire_delegation_routing_dispatcher(engine)

        call_kwargs = engine.register_dispatcher.call_args[1]
        assert (
            "omnibase-infra.delegation-routing-request" in call_kwargs["message_types"]
        )

    @pytest.mark.asyncio
    async def test_route_uses_command_category(self) -> None:
        from unittest.mock import MagicMock

        from omnibase_core.enums import EnumMessageCategory
        from omnibase_infra.models.dispatch.model_dispatch_route import (
            ModelDispatchRoute,
        )
        from omnibase_infra.nodes.node_delegation_orchestrator.wiring import (
            wire_delegation_routing_dispatcher,
        )

        engine = MagicMock()
        engine.register_dispatcher = MagicMock()
        engine.register_route = MagicMock()

        await wire_delegation_routing_dispatcher(engine)

        route: ModelDispatchRoute = engine.register_route.call_args[0][0]
        assert isinstance(route, ModelDispatchRoute)
        assert route.message_category == EnumMessageCategory.COMMAND
        assert route.message_type == "omnibase-infra.delegation-routing-request"
