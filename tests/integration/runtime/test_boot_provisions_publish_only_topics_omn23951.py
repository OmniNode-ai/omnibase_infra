# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot provisioning creates publish topics of unattached contracts (OMN-23951)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.boot_topic_provisioning import (
    provision_unattached_publish_topics,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)

PUBLISH_ONLY_TOPIC = "onex.evt.omnimarket.integration-floor-snapshot.v1"
INTERLEAVED_TOPIC = "onex.evt.platform.integration-interleaved-emitted.v1"


class _BrokerStandIn:
    """Topic provisioner holding the set of topics the 'broker' knows."""

    def __init__(self) -> None:
        self.topics: set[str] = set()

    async def ensure_topic_exists(self, *, topic_name: str, **_: object) -> bool:
        self.topics.add(topic_name)
        return True


def _contract(name: str, publish: tuple[str, ...]) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(subscribe_topics=(), publish_topics=publish),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_publish_only_topic_exists_after_boot_provisioning() -> None:
    broker = _BrokerStandIn()
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_publish_only", (PUBLISH_ONLY_TOPIC,)),
            _contract("node_interleaved", (INTERLEAVED_TOPIC,)),
        )
    )
    assert PUBLISH_ONLY_TOPIC not in broker.topics

    ensured = await provision_unattached_publish_topics(
        manifest, {"node_interleaved"}, broker, max_concurrent=2
    )

    assert ensured == {"node_publish_only": (PUBLISH_ONLY_TOPIC,)}
    assert broker.topics == {PUBLISH_ONLY_TOPIC}
