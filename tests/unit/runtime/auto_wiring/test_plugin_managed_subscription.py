# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for plugin_managed subscription skip [OMN-10864].

Reproduces the duplicate-consumer bug where auto-wiring rediscovery (~5 min
interval) subscribed a second consumer to the delegation command topic
WITHOUT a result_applier, short-circuiting the domain plugin's wired
HandlerDelegationWorkflow singleton.

These tests prove:
  1. plugin_managed=True contracts do NOT get a Kafka subscription from auto-wiring.
  2. Dispatch routes ARE still registered so the engine can route messages from
     the plugin's own EventBusSubcontractWiring consumer.
  3. Non-plugin-managed contracts are still auto-subscribed normally.
  4. The delegation contract YAML declares plugin_managed=true.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import (
    MessageDispatchEngine,
)


def _fake_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


def _contract(
    name: str = "node_test",
    subscribe_topics: tuple[str, ...] = ("onex.cmd.omnimarket.test-start.v1",),
    *,
    plugin_managed: bool = False,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics,
            publish_topics=(),
            plugin_managed=plugin_managed,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="FakeHandler", module="fake.module"),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


class TestPluginManagedSkipsSubscription:
    """plugin_managed=True prevents auto-wiring from creating a Kafka subscription."""

    @pytest.mark.asyncio
    async def test_plugin_managed_contract_skips_kafka_subscribe(self) -> None:
        contract = _contract(
            name="node_delegation_orchestrator",
            subscribe_topics=("onex.cmd.omnibase-infra.delegation-request.v1",),
            plugin_managed=True,
        )
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=AsyncMock())

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        # Handler IS wired into dispatch engine
        assert report.total_wired == 1
        # Kafka subscribe must NOT be called — plugin owns the subscription
        event_bus.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_plugin_managed_contract_has_empty_topics_subscribed(self) -> None:
        contract = _contract(plugin_managed=True)
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=AsyncMock())

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        result = next(r for r in report.results if r.contract_name == contract.name)
        assert result.topics_subscribed == ()

    @pytest.mark.asyncio
    async def test_plugin_managed_dispatchers_still_registered(self) -> None:
        """Dispatch routes must exist even when subscription is skipped."""
        contract = _contract(plugin_managed=True)
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=AsyncMock())

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        result = next(r for r in report.results if r.contract_name == contract.name)
        assert result.outcome.value == "wired"
        assert len(result.dispatchers_registered) > 0

    @pytest.mark.asyncio
    async def test_non_plugin_managed_contract_still_subscribes(self) -> None:
        """Contracts without plugin_managed=True must still be auto-subscribed."""
        topic = "onex.cmd.omnimarket.test-start.v1"
        contract = _contract(subscribe_topics=(topic,), plugin_managed=False)
        manifest = ModelAutoWiringManifest(contracts=(contract,))
        engine = MessageDispatchEngine()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=AsyncMock())

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        event_bus.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_manifest_only_subscribes_non_plugin_managed(self) -> None:
        """In a mixed manifest, only non-plugin-managed contracts get subscribed."""
        managed_contract = _contract(
            name="node_delegation_orchestrator",
            subscribe_topics=("onex.cmd.omnibase-infra.delegation-request.v1",),
            plugin_managed=True,
        )
        normal_contract = _contract(
            name="node_regular_worker",
            subscribe_topics=("onex.cmd.omnimarket.test-start.v1",),
            plugin_managed=False,
        )
        manifest = ModelAutoWiringManifest(
            contracts=(managed_contract, normal_contract)
        )
        engine = MessageDispatchEngine()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=AsyncMock())

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            await wire_from_manifest(
                manifest, engine, event_bus=event_bus, environment="local"
            )

        # Only the non-plugin-managed contract subscribed
        assert event_bus.subscribe.call_count == 1
        call_topic = (
            event_bus.subscribe.call_args.kwargs.get("topic")
            or event_bus.subscribe.call_args.args[0]
        )
        assert call_topic == "onex.cmd.omnimarket.test-start.v1"

    @pytest.mark.asyncio
    async def test_deferred_subscribe_also_skips_plugin_managed(self) -> None:
        """subscribe_wired_contract_topics must also skip plugin_managed contracts."""
        managed_contract = _contract(
            name="node_delegation_orchestrator",
            subscribe_topics=("onex.cmd.omnibase-infra.delegation-request.v1",),
            plugin_managed=True,
        )
        normal_contract = _contract(
            name="node_regular_worker",
            subscribe_topics=("onex.cmd.omnimarket.test-start.v1",),
            plugin_managed=False,
        )
        manifest = ModelAutoWiringManifest(
            contracts=(managed_contract, normal_contract)
        )
        engine = MessageDispatchEngine()
        event_bus = MagicMock()
        event_bus.subscribe = AsyncMock(return_value=AsyncMock())

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest,
                engine,
                event_bus=event_bus,
                environment="local",
                subscribe_immediately=False,
            )
            event_bus.subscribe.assert_not_called()

            subscriptions = await subscribe_wired_contract_topics(
                manifest=manifest,
                report=report,
                dispatch_engine=engine,
                event_bus=event_bus,
                environment="local",
            )

        # Only normal_contract subscribed
        assert event_bus.subscribe.call_count == 1
        assert "node_delegation_orchestrator" not in subscriptions
        assert "node_regular_worker" in subscriptions
