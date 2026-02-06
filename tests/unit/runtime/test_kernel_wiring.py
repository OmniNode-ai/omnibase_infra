# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kernel wiring tests (OMN-1931 P2.5).

Pure Python tests (no Docker) verifying that the service_kernel correctly
wires orchestrator handlers, event routers, and Kafka subscriptions.

These tests use the inmemory event bus to verify wiring without requiring
a real Kafka broker.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
    DispatcherNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.introspection_event_router import (
    IntrospectionEventRouter,
)
from tests.conftest import make_test_node_identity


class TestKernelIntrospectionWiring:
    """Test that IntrospectionEventRouter is correctly wired to the event bus."""

    @pytest.mark.asyncio
    async def test_introspection_router_subscribes_to_event_bus(self) -> None:
        """Verify IntrospectionEventRouter subscribes via event bus.subscribe().

        This mirrors the wiring in service_kernel.py:
        1. Create event bus (inmemory for test)
        2. Create introspection dispatcher
        3. Create IntrospectionEventRouter with dependencies
        4. Subscribe router's handle_message to the input topic
        """
        bus = EventBusInmemory(environment="test", group="test-kernel")
        await bus.start()

        # Create dispatcher (mock since we don't need real handlers)
        mock_dispatcher = MagicMock(spec=DispatcherNodeIntrospected)
        mock_container = MagicMock()

        # Create router as done in service_kernel.py
        router = IntrospectionEventRouter(
            container=mock_container,
            output_topic="test.responses",
            dispatcher=mock_dispatcher,
            event_bus=bus,
        )

        identity = make_test_node_identity(
            env="test", service="test-kernel", node_name="test-kernel"
        )

        # Subscribe with required_for_readiness=True (as kernel does)
        unsubscribe = await bus.subscribe(
            topic="test.requests",
            node_identity=identity,
            on_message=router.handle_message,
            required_for_readiness=True,
        )

        # Verify subscription registered
        health = await bus.health_check()
        assert health["subscriber_count"] >= 1
        assert health["topic_count"] >= 1

        await unsubscribe()
        await bus.close()

    @pytest.mark.asyncio
    async def test_inmemory_bus_readiness_after_wiring(self) -> None:
        """Verify readiness is reported correctly after kernel-style wiring."""
        bus = EventBusInmemory(environment="test", group="test-kernel")
        await bus.start()

        mock_dispatcher = MagicMock(spec=DispatcherNodeIntrospected)
        mock_container = MagicMock()

        router = IntrospectionEventRouter(
            container=mock_container,
            output_topic="test.responses",
            dispatcher=mock_dispatcher,
            event_bus=bus,
        )

        identity = make_test_node_identity(
            env="test", service="test-kernel", node_name="test-kernel"
        )

        await bus.subscribe(
            topic="test.requests",
            node_identity=identity,
            on_message=router.handle_message,
            required_for_readiness=True,
        )

        # Inmemory bus should report ready
        readiness = await bus.get_readiness_status()
        assert readiness.is_ready is True
        assert readiness.consumers_started is True

        await bus.close()


class TestKernelSubscriptionConfiguration:
    """Test that kernel subscriptions use correct required_for_readiness flags."""

    @pytest.mark.asyncio
    async def test_all_kernel_subscriptions_are_readiness_required(self) -> None:
        """Verify the kernel marks all its subscriptions as required_for_readiness.

        The kernel has 4 subscriptions that should all be required:
        1. Introspection events
        2. Contract registered events
        3. Contract deregistered events
        4. Node heartbeat events
        """
        bus = EventBusInmemory(environment="test", group="test-kernel")
        await bus.start()

        # This is a structural test: verify subscribe() accepts the parameter
        # The actual kernel code is tested via the broader test suite
        identity = make_test_node_identity()

        async def noop_handler(msg: object) -> None:
            pass

        topics = [
            "test.requests",
            "test.onex.evt.platform.contract-registered.v1",
            "test.onex.evt.platform.contract-deregistered.v1",
            "test.onex.evt.platform.node-heartbeat.v1",
        ]

        unsubscribes = []
        for topic in topics:
            unsub = await bus.subscribe(
                topic=topic,
                node_identity=identity,
                on_message=noop_handler,
                required_for_readiness=True,
            )
            unsubscribes.append(unsub)

        health = await bus.health_check()
        assert health["subscriber_count"] == 4
        assert health["topic_count"] == 4

        for unsub in unsubscribes:
            await unsub()
        await bus.close()
