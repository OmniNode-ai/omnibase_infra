# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21950: the forwarder's lane-mirror source bus comes from the container.

Every runtime has a bus: the in-memory bus by default, a hosted bus where its
overlay says so. The forwarder runs inside the runtime it forwards for, so its
source leg is that runtime's own ``ProtocolEventBus`` -- resolved from the
container, chosen by the overlay -- and never an adapter the node constructs.

Before this change ``ModelGatewayForwarderRuntimeConfig`` typed the lane-mirror
source leg as ``ModelKafkaEventBusConfig``, so an overlay could not select the
runtime's in-memory bus at all: constructing the config with an in-memory source
raised. That raise is the positive control for these tests (RED on the base
commit).
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderConfig,
    ModelGatewayLaneMirrorConfig,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_runtime_config import (
    ModelGatewayForwarderRuntimeConfig,
)

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnibase-infra.ledger-append.v1"
_PARENT_BUS = "parent-redpanda:9092"


def _forwarder() -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=UUID("79afa726-3852-464f-b7a4-d4b8b9c75ee7"),
            tenant_slug="runtime-a",
            principal_id="t-79afa7263852464fb7a4d4b8b9c75ee7",
        ),
        local_transport_flavor="containerized",
        dedupe_store_path=Path("/app/data/lane-mirror.sqlite3"),
        lane_mirror=ModelGatewayLaneMirrorConfig(
            source_lane="runtime",
            mirror_lanes=("parent",),
            declared_lanes=("runtime", "parent"),
            topics=(_COMMAND_TOPIC,),
        ),
    )


def _parent_leg() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers=_PARENT_BUS,
        environment="lane-mirror-parent",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )


def test_runtime_config_accepts_in_memory_source_from_source_bus_container() -> None:
    """The overlay selects the runtime's own bus; the config names no adapter.

    RED on the base commit: the source leg was typed ``ModelKafkaEventBusConfig``
    so ``{"source": "container"}`` failed validation.
    """
    config = ModelGatewayForwarderRuntimeConfig.model_validate(
        {
            "forwarder": _forwarder(),
            "lane_mirror_source_bus": {"source": "container"},
            "lane_mirror_buses": {"parent": _parent_leg()},
        }
    )
    source = config.lane_mirror_source_bus
    assert source is not None
    assert not isinstance(source, ModelKafkaEventBusConfig)
    assert config.lane_mirror_source_from_container is True


def test_runtime_config_keeps_kafka_in_memory_source_validation_for_kafka_legs() -> (
    None
):
    """A Kafka-selected source is unchanged: auto-commit is still refused."""
    with pytest.raises(ValueError, match="enable_auto_commit=false"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=_forwarder(),
            lane_mirror_source_bus=ModelKafkaEventBusConfig(
                bootstrap_servers="runtime-redpanda:9092",
                environment="lane-mirror-source",
                enable_auto_commit=True,
                auto_offset_reset="earliest",
            ),
            lane_mirror_buses={"parent": _parent_leg()},
        )


def test_kafka_source_is_not_from_source_bus_container() -> None:
    """Changing only the overlay changes the bus: a Kafka leg is not container."""
    config = ModelGatewayForwarderRuntimeConfig(
        forwarder=_forwarder(),
        lane_mirror_source_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="runtime-redpanda:9092",
            environment="lane-mirror-source",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        ),
        lane_mirror_buses={"parent": _parent_leg()},
    )
    assert config.lane_mirror_source_from_container is False


def test_in_memory_source_refuses_an_unknown_container_selector() -> None:
    """Only the container selector is accepted; nothing else is guessed."""
    with pytest.raises(ValueError):
        ModelGatewayForwarderRuntimeConfig.model_validate(
            {
                "forwarder": _forwarder(),
                "lane_mirror_source_bus": {"source": "inmemory-adapter"},
                "lane_mirror_buses": {"parent": _parent_leg()},
            }
        )
