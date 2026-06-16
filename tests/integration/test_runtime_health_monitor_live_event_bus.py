# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for live EventBus-backed runtime health expectations.

Integration Test Coverage gate: OMN-9648.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    ServiceRuntimeHealthMonitor,
)


def _contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_legacy_projection",
        node_type="PROJECTION",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name="node_legacy_projection",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.omnimarket.legacy-projection.v1",),
            publish_topics=(),
        ),
    )


@pytest.mark.asyncio
async def test_monitor_uses_live_event_bus_groups_for_runtime_liveness() -> None:
    """Stale discovered contracts and empty broker groups do not degrade runtime."""
    contract = _contract()
    topic = contract.event_bus.subscribe_topics[0]
    live_group = f"runtime.projection.consume.v1.__i.main.__t.{topic}"
    stale_empty_group = f"old.projection.consume.v1.__i.main.__t.{topic}"
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.get_consumer_groups.return_value = {
        (topic, "runtime.projection.consume.v1"): live_group
    }
    monitor = ServiceRuntimeHealthMonitor(
        event_bus=bus,
        bootstrap_servers="localhost:9092",
    )

    with (
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=ModelAutoWiringManifest(contracts=(contract,), errors=()),
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
            return_value=[
                ConsumerGroupSnapshot(group_id=live_group, state="STABLE"),
                ConsumerGroupSnapshot(group_id=stale_empty_group, state="EMPTY"),
            ],
        ),
    ):
        event = await monitor.run_once()

    assert event.status == "HEALTHY"
    assert event.discovery_error_count == 0
    assert event.empty_consumer_group_count == 0
    assert event.uncovered_topic_count == 0
