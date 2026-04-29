# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime health exact consumer group matching."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    ServiceRuntimeHealthMonitor,
    _expected_consumer_groups_from_manifest,
)


def _contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_session_orchestrator",
        node_type="ORCHESTRATOR",
        contract_version=ModelContractVersion(major=1, minor=2, patch=3),
        contract_path=__file__,
        entry_point_name="node_session_orchestrator",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.omnimarket.session-orchestrator-start.v1",),
            publish_topics=(),
        ),
    )


@pytest.mark.asyncio
async def test_runtime_health_requires_exact_event_bus_consumer_group() -> None:
    """Topic-shaped but wrong groups must not satisfy runtime coverage."""
    manifest = ModelAutoWiringManifest(contracts=(_contract(),), errors=())
    expected = _expected_consumer_groups_from_manifest(manifest)[0]
    wrong_topic_shaped_group = f"wrong-prefix.__t.{expected.topic}"
    monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="redpanda:9092")

    with (
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
            return_value=[
                ConsumerGroupSnapshot(
                    group_id=wrong_topic_shaped_group,
                    state="STABLE",
                ),
            ],
        ),
    ):
        event = await monitor.run_once()

    assert event.uncovered_topic_count == 1
    assert event.status == "DEGRADED"
    topic_coverage = next(d for d in event.dimensions if d.name == "topic_coverage")
    assert expected.group_id in topic_coverage.detail


@pytest.mark.asyncio
async def test_runtime_health_accepts_exact_event_bus_consumer_group() -> None:
    """The exact EventBusKafka group ID satisfies runtime topic coverage."""
    manifest = ModelAutoWiringManifest(contracts=(_contract(),), errors=())
    expected = _expected_consumer_groups_from_manifest(manifest)[0]
    monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="redpanda:9092")

    with (
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
            return_value=[
                ConsumerGroupSnapshot(group_id=expected.group_id, state="STABLE"),
            ],
        ),
    ):
        event = await monitor.run_once()

    assert event.uncovered_topic_count == 0
    assert event.status == "HEALTHY"
