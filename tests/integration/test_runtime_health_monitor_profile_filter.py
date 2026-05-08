# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime health monitor profile ownership."""

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


def _contract(name: str, topic: str, profile: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name="omnimarket",
        runtime_profiles=(profile,),
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
    )


@pytest.mark.asyncio
async def test_runtime_health_monitor_uses_profile_filtered_contracts() -> None:
    main_contract = _contract(
        "node_main_profile",
        "onex.cmd.omnimarket.main-profile.v1",
        "main",
    )
    effects_contract = _contract(
        "node_effects_profile",
        "onex.cmd.omnimarket.effects-profile.v1",
        "effects",
    )
    manifest = ModelAutoWiringManifest(
        contracts=(main_contract, effects_contract),
        errors=(),
    )
    expected = _expected_consumer_groups_from_manifest(
        ModelAutoWiringManifest(contracts=(main_contract,), errors=())
    )[0]
    monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="redpanda:9092")

    with (
        patch.dict("os.environ", {"RUNTIME_PROFILE": "main"}),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
            return_value=manifest,
        ),
        patch(
            "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
            return_value=[
                ConsumerGroupSnapshot(group_id=expected.group_id, state="STABLE")
            ],
        ),
    ):
        event = await monitor.run_once()

    assert event.contract_count == 1
    assert event.uncovered_topic_count == 0
    assert event.status == "HEALTHY"
