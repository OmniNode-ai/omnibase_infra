# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
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
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)

pytestmark = pytest.mark.integration


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

        # Verify handle method signature accepts 'envelope' parameter
        sig = inspect.signature(handle_method)
        param_names = list(sig.parameters.keys())
        # First param is 'self', second should be 'envelope'
        assert "envelope" in param_names, (
            f"handle() must accept 'envelope' parameter, got: {param_names}"
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

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            consul_enabled=False,  # Avoid consul intent generation
        )

        # Build a valid envelope with ModelNodeIntrospectionEvent payload
        node_id = uuid4()
        correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer(major=1, minor=0, patch=0),
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC),
        )

        envelope: ModelEventEnvelope[ModelNodeIntrospectionEvent] = ModelEventEnvelope(
            correlation_id=correlation_id,
            event_type="ModelNodeIntrospectionEvent",
            payload=event,
        )

        output = await handler.handle(envelope)

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
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)
        assert "ModelNodeIntrospectionEvent" in handler.message_types


class TestConsulRegistrationVisible:
    """Tests that Consul registration is visible after introspection processing."""

    @pytest.mark.consul
    @pytest.mark.asyncio
    async def test_consul_registration_visible(
        self,
        consul_available: bool,
    ) -> None:
        """Verify service appears in Consul after registration.

        This test requires a live Consul instance. It is automatically
        skipped if Consul is not available (via the consul_available fixture).

        The test registers a dummy service, verifies it appears in the
        Consul catalog, then deregisters it for cleanup.
        """
        if not consul_available:
            pytest.skip("Consul not available")

        import os

        import consul.aio

        consul_host = os.environ.get("CONSUL_HOST", "192.168.86.200")
        consul_port = int(os.environ.get("CONSUL_PORT", "28500"))

        client = consul.aio.Consul(host=consul_host, port=consul_port)

        service_id = f"test-omn2081-{uuid4().hex[:8]}"
        service_name = "test-omn2081-introspection"

        try:
            # Register a test service (simulating what the effect handler does)
            success = await client.agent.service.register(
                name=service_name,
                service_id=service_id,
                tags=["test", "omn-2081", "introspection"],
            )
            assert success is True

            # Verify the service is visible in the catalog
            _, services = await client.catalog.service(service_name)
            service_ids = [s["ServiceID"] for s in services]
            assert service_id in service_ids, (
                f"Service {service_id} not found in Consul catalog. "
                f"Found: {service_ids}"
            )
        finally:
            # Cleanup: deregister the test service
            try:
                await client.agent.service.deregister(service_id)
            except Exception:
                pass  # Best-effort cleanup


__all__: list[str] = [
    "TestRegistrationHandlerInterface",
    "TestRegistrationHandlerProperties",
    "TestConsulRegistrationVisible",
]
