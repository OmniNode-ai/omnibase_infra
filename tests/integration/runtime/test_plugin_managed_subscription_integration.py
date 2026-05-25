# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests: plugin_managed=True prevents duplicate Kafka subscriptions.

These tests verify end-to-end that the plugin_managed flag correctly prevents
auto-wiring from creating duplicate consumer groups for domain-plugin-owned topics
(OMN-10864 regression guard).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import (
    MessageDispatchEngine,
)

pytestmark = pytest.mark.integration

_DELEGATION_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
_NORMAL_TOPIC = "onex.cmd.omnimarket.normal-worker.v1"


class _FakeHandler:
    async def handle(self, envelope: object) -> None:
        return None


def _make_contract(
    name: str,
    topic: str,
    *,
    plugin_managed: bool = False,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/tmp/integration-pm-test/{name}/contract.yaml"),  # noqa: S108
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=(),
            plugin_managed=plugin_managed,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="_FakeHandler", module=__name__),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_plugin_managed_prevents_duplicate_subscription_immediate() -> None:
    """wire_from_manifest must NOT call event_bus.subscribe for plugin_managed contracts."""
    managed = _make_contract(
        "node_delegation_orchestrator", _DELEGATION_TOPIC, plugin_managed=True
    )
    manifest = ModelAutoWiringManifest(contracts=(managed,), errors=())
    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    assert report.total_wired == 1
    event_bus.subscribe.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_manifest_only_subscribes_non_plugin_managed_integration() -> None:
    """Mixed manifest: plugin_managed contract skipped; normal contract subscribed."""
    managed = _make_contract(
        "node_delegation_orchestrator", _DELEGATION_TOPIC, plugin_managed=True
    )
    normal = _make_contract("node_normal_worker", _NORMAL_TOPIC, plugin_managed=False)
    manifest = ModelAutoWiringManifest(contracts=(managed, normal), errors=())
    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    assert report.total_wired == 2
    # Exactly one subscription: only the normal (non-plugin-managed) contract
    assert event_bus.subscribe.call_count == 1
    call_topic = (
        event_bus.subscribe.call_args.kwargs.get("topic")
        or event_bus.subscribe.call_args.args[0]
    )
    assert call_topic == _NORMAL_TOPIC


@pytest.mark.asyncio
async def test_deferred_subscription_skips_plugin_managed_integration() -> None:
    """Deferred subscribe_wired_contract_topics also skips plugin_managed contracts."""
    managed = _make_contract(
        "node_delegation_orchestrator", _DELEGATION_TOPIC, plugin_managed=True
    )
    normal = _make_contract("node_normal_worker", _NORMAL_TOPIC, plugin_managed=False)
    manifest = ModelAutoWiringManifest(contracts=(managed, normal), errors=())
    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
            subscribe_immediately=False,
        )
        event_bus.subscribe.assert_not_called()

        dispatch_engine.freeze()

        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    assert event_bus.subscribe.call_count == 1
    assert "node_delegation_orchestrator" not in subscriptions
    assert "node_normal_worker" in subscriptions
    assert subscriptions["node_normal_worker"] == (_NORMAL_TOPIC,)
