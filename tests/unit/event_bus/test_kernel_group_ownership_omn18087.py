# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The writer and the main runtime resolve different kernel groups in ``EventBusKafka``.

Resolves the effective consumer group each process joins, through
``EventBusKafka._resolve_effective_group_id`` (the one place the bus composes the
joined group, with no public accessor) under the instance id both containers are
deployed with (``runtime-main``), for every consumer the kernel builds from the
node selection. No broker is contacted, so this is a unit test of the naming: two
processes that resolve the same name are two members of one group on a
one-partition topic (OMN-18087). Membership on a real broker is a slot reading,
not this test.
"""

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.runtime import ModelRuntimeNodeSelection
from omnibase_infra.utils import compute_consumer_group_id

INSTANCE_ID = "runtime-main"
CONFIG_NAME = "runtime_config"
TOPIC = "onex.evt.platform.contract-registered.v1"
KERNEL_NODES = (
    "contract-registry",
    "node_savings_estimation_compute",
    "runtime-error-triage",
    CONFIG_NAME,
)
HOOK_EVENTS_CONTRACT = "projection_claude_hook_events"


def _bus() -> EventBusKafka:
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="dev",
            instance_id=INSTANCE_ID,
        )
    )


def _effective_group(bus: EventBusKafka, identity: ModelNodeIdentity) -> str:
    group = compute_consumer_group_id(identity, EnumConsumerGroupPurpose.CONSUME)
    return bus._resolve_effective_group_id(group, TOPIC, uuid4(), (TOPIC, group))


@pytest.mark.parametrize("node_name", KERNEL_NODES)
def test_writer_and_main_resolve_different_kernel_groups(node_name: str) -> None:
    main = ModelRuntimeNodeSelection.from_env({})
    writer = ModelRuntimeNodeSelection.from_env(
        {"ONEX_RUNTIME_NODE_SELECTION": HOOK_EVENTS_CONTRACT}
    )
    bus = _bus()
    main_group = _effective_group(
        bus, main.kernel_identity("dev", CONFIG_NAME, node_name, "1.0.0")
    )
    writer_group = _effective_group(
        bus, writer.kernel_identity("dev", CONFIG_NAME, node_name, "1.0.0")
    )
    assert main_group != writer_group
    assert main_group.endswith(f".__i.{INSTANCE_ID}.__t.{TOPIC}")
    assert writer_group.endswith(f".__i.{INSTANCE_ID}.__t.{TOPIC}")


def test_the_selected_contracts_own_group_still_carries_the_instance_id() -> None:
    """Its identity is the contract's, so the committed offsets stay valid."""
    contract_identity = ModelNodeIdentity(
        env="dev",
        service="omnimarket",
        node_name=HOOK_EVENTS_CONTRACT,
        version="1.0.0",
    )
    group = _effective_group(_bus(), contract_identity)
    assert f".__i.{INSTANCE_ID}." in group
    assert HOOK_EVENTS_CONTRACT in group
