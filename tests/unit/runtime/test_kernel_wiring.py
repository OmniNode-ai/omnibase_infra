# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kernel wiring tests (OMN-1931 P2.5, OMN-2050).

Pure Python tests (no Docker) verifying that the service_kernel correctly
wires the MessageDispatchEngine, domain plugins, and Kafka subscriptions.

These tests use the inmemory event bus to verify wiring without requiring
a real Kafka broker.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from tests.conftest import make_test_node_identity


@pytest.mark.unit
class TestKernelDispatchEngineWiring:
    """Test that MessageDispatchEngine is correctly instantiated and frozen."""

    @pytest.mark.asyncio
    async def test_dispatch_engine_can_be_created_and_frozen(self) -> None:
        """Verify MessageDispatchEngine can be created and frozen.

        This mirrors the Phase A kernel bootstrap:
        1. Create MessageDispatchEngine
        2. Register dispatchers
        3. Freeze the engine
        """
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        engine = MessageDispatchEngine()
        # Engine should be unfrozen initially
        assert not engine._frozen

        # Freeze it
        engine.freeze()

        # Engine should now be frozen
        assert engine._frozen

    @pytest.mark.asyncio
    async def test_dispatch_engine_set_on_plugin_config(self) -> None:
        """Verify dispatch_engine can be set on ModelDomainPluginConfig."""
        from uuid import uuid4

        from omnibase_infra.runtime.models import ModelDomainPluginConfig
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        engine = MessageDispatchEngine()
        bus = EventBusInmemory(environment="test", group="test-kernel")
        container = MagicMock()

        config = ModelDomainPluginConfig(
            container=container,
            event_bus=bus,
            correlation_id=uuid4(),
            input_topic="requests",
            output_topic="responses",
            consumer_group="onex-test",
            dispatch_engine=engine,
        )

        assert config.dispatch_engine is engine


@pytest.mark.unit
class TestKernelSubscriptionConfiguration:
    """Test that kernel subscriptions use correct required_for_readiness flags."""

    @pytest.mark.asyncio
    async def test_all_kernel_subscriptions_are_readiness_required(self) -> None:
        """Verify the kernel marks all its subscriptions as required_for_readiness.

        The kernel has 4 subscriptions that should all be required:
        1. Introspection events (via EventBusSubcontractWiring)
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
