# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18934: a contract-declared bound is recorded on the real in-memory bus.

One run of the real stack: a contract YAML on disk, the real
``wire_from_manifest`` and a real ``EventBusInmemory``. Nothing is substituted
except handler class import. Readback goes through the public snapshot.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import ModelHandlerRef
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.utils import compute_consumer_group_id

pytestmark = pytest.mark.integration

SUBSCRIBE_TOPIC = "onex.cmd.omnibase-infra.omn18934-inference-request.v1"
PUBLISH_TOPIC = "onex.evt.omnibase-infra.omn18934-inference-response.v1"
DECLARED_BOUND = 3

_CONTRACT = f"""
name: node_omn18934_effect
node_type: EFFECT_GENERIC
contract_version: 1.0.0
description: fixture
event_bus:
  subscribe_topics:
    - {SUBSCRIBE_TOPIC}
  publish_topics:
    - {PUBLISH_TOPIC}
consume_concurrency:
  max_in_flight_records: {DECLARED_BOUND}
"""


class _Handler:
    async def handle(self, request: object) -> object:
        return request


@pytest.mark.asyncio
async def test_contract_bound_is_recorded_on_the_inmemory_bus(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CONTRACT, encoding="utf-8")
    contract = ModelDiscoveredContract(
        name="node_omn18934_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_omn18934_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(SUBSCRIBE_TOPIC,),
            publish_topics=(PUBLISH_TOPIC,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="_Handler", module=__name__),
                    event_model=None,
                ),
            ),
        ),
    )
    bus = EventBusInmemory(environment="test", group="omn18934-integration")

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_Handler,
    ):
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(contract,)),
            MessageDispatchEngine(),
            event_bus=bus,
            environment="local",
        )

    group_id = compute_consumer_group_id(
        ModelNodeIdentity(
            env="local",
            service=contract.package_name,
            node_name=contract.name,
            version=str(contract.contract_version),
        )
    )
    assert bus.consume_concurrency == {(SUBSCRIBE_TOPIC, group_id): DECLARED_BOUND}
