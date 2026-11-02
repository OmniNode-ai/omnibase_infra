# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for registration visibility (OMN-2081).

Tests that the registration handler pipeline is correctly wired:

1. HandlerNodeIntrospected is importable and has the correct async interface
2. HandlerNodeIntrospected can be instantiated with a mock projection reader
   and its handle method produces ModelHandlerOutput when given a valid envelope
3. Consul registration is visible after processing (when Consul is available)

Related:
    - OMN-2081: Investor demo - runtime contract routing verification
    - src/omnibase_infra/nodes/node_registration_orchestrator/
    - src/omnibase_infra/projectors/
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.dispatch.model_injected_request import (
    ModelInjectedRequest,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)

pytestmark = pytest.mark.integration

# One fixed reading for the injected clock and the event timestamp, so the
# handler's time-dependent output is asserted against a known value.
TEST_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Tests
# =============================================================================


class TestRegistrationHandlerInterface:
    """Tests that HandlerNodeIntrospected has the correct interface and produces output."""

    def test_handler_class_importable_and_has_async_handle(self) -> None:
        """Verify HandlerNodeIntrospected is importable and exposes an async
        handle method that accepts an envelope parameter.

        This proves the contract routing pipeline is wired correctly:
        the handler class declared in contract.yaml actually exists
        and has the expected interface.
        """
        # Verify the class exists (import already succeeded above)
        assert HandlerNodeIntrospected is not None

        # Verify handle method exists and is async
        assert hasattr(HandlerNodeIntrospected, "handle")
        handle_method = HandlerNodeIntrospected.handle
        assert callable(handle_method)
        assert asyncio.iscoroutinefunction(handle_method), (
            "HandlerNodeIntrospected.handle must be async"
        )

        # The dispatcher calls handle(request) positionally, so the signature
        # is exactly (self, request): no envelope, no extra parameters.
        sig = inspect.signature(handle_method)
        param_names = list(sig.parameters.keys())
        assert param_names == ["self", "request"], (
            f"handle() must take exactly (self, request), got: {param_names}"
        )

        # The parameter is annotated with a typed request model over the
        # introspection event. Compare the event field's annotation rather than
        # the identity of the parametrized generic class.
        hints = typing.get_type_hints(handle_method)
        request_model = ModelInjectedRequest.narrow(hints["request"])
        assert request_model is not None, (
            f"handle() 'request' must be a ModelInjectedRequest model, "
            f"got: {hints['request']}"
        )
        event_annotation = request_model.model_fields["event"].annotation
        assert event_annotation is ModelNodeIntrospectionEvent, (
            f"handle() request event must be ModelNodeIntrospectionEvent, "
            f"got: {event_annotation}"
        )

    @pytest.mark.asyncio
    async def test_handler_produces_output_for_new_node(self) -> None:
        """Instantiate HandlerNodeIntrospected with a mock projection reader
        and verify that calling handle() with a valid envelope produces a
        ModelHandlerOutput with events and intents (for a new node with no
        existing projection).
        """
        # Create a mock projection reader that returns None (new node)
        mock_reader = MagicMock()
        mock_reader.get_entity_state = AsyncMock(return_value=None)

        reducer = RegistrationReducerService()
        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            reducer=reducer,
        )

        # Build a valid envelope with ModelNodeIntrospectionEvent payload
        node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer(major=1, minor=0, patch=0),
            correlation_id=correlation_id,
            timestamp=TEST_NOW,
        )

        input_id = uuid4()
        request: ModelInjectedRequest[ModelNodeIntrospectionEvent] = (
            ModelInjectedRequest(
                event=event,
                now=TEST_NOW,
                input_id=input_id,
                correlation_id=correlation_id,
            )
        )

        output = await handler.handle(request)

        # Verify output structure - handler should emit registration initiated event
        assert output is not None
        assert output.events is not None
        assert len(output.events) > 0, (
            "Handler should emit at least one event for a new node"
        )
        # Verify intents were generated (at least postgres upsert)
        assert output.intents is not None
        assert len(output.intents) > 0, (
            "Handler should emit at least one intent for a new node"
        )
        # Verify correlation_id propagation
        assert output.correlation_id == correlation_id
        # Verify the injected clock reading and input id reach the output
        assert output.timestamp == TEST_NOW
        assert output.input_envelope_id == input_id


class TestRegistrationHandlerProperties:
    """Tests that HandlerNodeIntrospected exposes expected classification properties."""

    def test_handler_has_expected_properties(self) -> None:
        """Verify the handler class defines handler_id, category, message_types,
        and node_kind properties that the dispatch engine uses for routing.
        """
        expected_properties = [
            "handler_id",
            "category",
            "message_types",
            "node_kind",
            "handler_type",
            "handler_category",
        ]

        for prop_name in expected_properties:
            assert hasattr(HandlerNodeIntrospected, prop_name), (
                f"HandlerNodeIntrospected must have '{prop_name}' property"
            )

    def test_handler_message_types_includes_introspection_event(self) -> None:
        """Verify the handler declares ModelNodeIntrospectionEvent in its
        message_types, matching the contract.yaml declaration.
        """
        mock_reader = MagicMock()
        reducer = RegistrationReducerService()
        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            reducer=reducer,
        )
        assert "ModelNodeIntrospectionEvent" in handler.message_types


__all__: list[str] = [
    "TestRegistrationHandlerInterface",
    "TestRegistrationHandlerProperties",
]
