# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for single-authority dispatcher wiring in registration plugin.

Verifies that ServiceRegistration owns explicit dispatcher adapters while
generic contract-driven auto-wiring owns event-bus subscriptions.

The duplicate-registration error (ONEX_CORE_064_DUPLICATE_REGISTRATION)
arose because both the legacy explicit path and the generic contract-driven
path registered dispatchers for the same contract.  This integration suite
confirms that:

1. The contract.yaml has the fields that justify auto-wiring subscription ownership.
2. The plugin can be imported and instantiated without side effects.
3. Dispatcher adapter registration happens through the domain wiring helper.

Fixtures from conftest.py:
    contract_data: parsed YAML content of node_registration_orchestrator/contract.yaml
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.integration


class TestContractDeclaresSubscribeTopics:
    """The contract must declare event_bus.subscribe_topics.

    The runtime reads event_bus.subscribe_topics to wire consumer subscriptions.
    Generic auto-wiring owns subscription setup for this node because the
    contract fully declares this field.
    """

    def test_event_bus_field_exists(self, contract_data: dict) -> None:
        """contract.yaml must have an event_bus section."""
        assert "event_bus" in contract_data, (
            "event_bus section is missing from contract.yaml — "
            "generic auto-wiring cannot own dispatcher registration without it"
        )

    def test_subscribe_topics_field_exists(self, contract_data: dict) -> None:
        """contract.yaml event_bus must have a subscribe_topics list."""
        event_bus = contract_data.get("event_bus", {})
        assert "subscribe_topics" in event_bus, (
            "event_bus.subscribe_topics is missing from contract.yaml — "
            "generic auto-wiring cannot own dispatcher registration without it"
        )

    def test_subscribe_topics_is_non_empty(self, contract_data: dict) -> None:
        """event_bus.subscribe_topics must contain at least one topic string."""
        topics = contract_data.get("event_bus", {}).get("subscribe_topics", [])
        assert isinstance(topics, list) and len(topics) > 0, (
            "event_bus.subscribe_topics must be a non-empty list — "
            "auto-wiring defers when no subscribe topics are declared"
        )

    def test_subscribe_topics_follow_naming_convention(
        self, contract_data: dict
    ) -> None:
        """Each subscribe topic must follow the onex.{cmd|evt}.{service}.{event}.v{N} pattern."""
        topics = contract_data.get("event_bus", {}).get("subscribe_topics", [])
        for topic in topics:
            assert topic.startswith("onex."), (
                f"Topic '{topic}' does not follow the onex.{{cmd|evt}}.*.v{{N}} convention"
            )


class TestContractDeclaresHandlerRouting:
    """The contract must declare handler_routing — auto-wiring reads this to build routes."""

    def test_handler_routing_field_exists(self, contract_data: dict) -> None:
        """contract.yaml must have a handler_routing section."""
        assert "handler_routing" in contract_data, (
            "handler_routing is missing from contract.yaml — "
            "auto-wiring cannot build routes without this section"
        )

    def test_handler_routing_has_handlers(self, contract_data: dict) -> None:
        """handler_routing.handlers must be a non-empty list."""
        routing = contract_data.get("handler_routing", {})
        handlers = routing.get("handlers", [])
        assert isinstance(handlers, list) and len(handlers) > 0, (
            "handler_routing.handlers must be a non-empty list"
        )


class TestPluginWireDispatchersNoDuplicateRegistration:
    """The plugin.wire_dispatchers() delegates to explicit dispatcher adapters."""

    def test_plugin_importable(self) -> None:
        """ServiceRegistration can be imported without side effects."""
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            ServiceRegistration,
        )

        assert ServiceRegistration is not None

    def test_plugin_instantiable(self) -> None:
        """ServiceRegistration() can be created without arguments or side effects."""
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            ServiceRegistration,
        )

        plugin = ServiceRegistration()
        assert plugin is not None
        assert plugin.plugin_id == "registration"

    @pytest.mark.asyncio
    async def test_wire_dispatchers_delegates_to_domain_wiring_helper(self) -> None:
        """wire_dispatchers() delegates adapter registration to domain wiring.

        Generic auto-wiring still owns subscriptions. The registration plugin
        owns the adapter layer because registration handlers expect domain
        envelope semantics rather than raw dict envelopes.
        """
        from dataclasses import dataclass
        from uuid import uuid4

        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            ServiceRegistration,
        )
        from omnibase_infra.runtime.models.model_domain_plugin_config import (
            ModelDomainPluginConfig,
        )

        @dataclass
        class _NodeIdentity:
            env: str = "test"
            service: str = "test-service"
            node_name: str = "test-service"
            version: str = "v1"

        engine = MagicMock()
        engine.register_dispatcher = MagicMock()
        engine.register_route = MagicMock()

        config = ModelDomainPluginConfig(
            container=MagicMock(),
            event_bus=MagicMock(),
            correlation_id=uuid4(),
            input_topic="onex.evt.platform.node-introspection.v1",
            output_topic="onex.evt.platform.node-registration-accepted.v1",
            consumer_group="onex-runtime-integration-test",
            dispatch_engine=engine,
            node_identity=_NodeIdentity(),  # type: ignore[arg-type]
        )

        plugin = ServiceRegistration()
        with patch(
            "omnibase_infra.nodes.node_registration_orchestrator.wiring"
            ".wire_registration_dispatchers",
            new=AsyncMock(
                return_value={
                    "dispatchers": ["dispatcher.registration.catalog-request"],
                    "routes": ["route.registration.catalog-request"],
                    "status": "success",
                }
            ),
        ) as wire_registration_dispatchers:
            result = await plugin.wire_dispatchers(config)

        assert result.success is True
        assert result.message == "Registration dispatchers wired"
        assert result.services_registered == ["dispatcher.registration.catalog-request"]
        wire_registration_dispatchers.assert_awaited_once_with(
            config.container,
            config.dispatch_engine,
            correlation_id=config.correlation_id,
            event_bus=config.event_bus,
        )
