# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for deferred auto-wiring subscription startup."""

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
from omnibase_infra.runtime.service_message_dispatch_engine import (
    MessageDispatchEngine,
)

pytestmark = pytest.mark.integration


def _contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_deferred_subscription_probe",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/tmp/deferred-subscription/contract.yaml"),  # noqa: S108
        entry_point_name="node_deferred_subscription_probe",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.omnimarket.deferred-subscription-probe.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerDeferredSubscriptionProbe",
                        module=__name__,
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


class HandlerDeferredSubscriptionProbe:
    async def handle(self, envelope: object) -> None:
        return None


@pytest.mark.asyncio
async def test_auto_wiring_defers_subscriptions_until_explicit_commit() -> None:
    contract = _contract()
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerDeferredSubscriptionProbe,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
            subscribe_immediately=False,
        )

        assert report.total_wired == 1
        event_bus.subscribe.assert_not_called()

        dispatch_engine.freeze()

        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    event_bus.subscribe.assert_called_once()
    assert subscriptions == {
        "node_deferred_subscription_probe": (
            "onex.cmd.omnimarket.deferred-subscription-probe.v1",
        )
    }
