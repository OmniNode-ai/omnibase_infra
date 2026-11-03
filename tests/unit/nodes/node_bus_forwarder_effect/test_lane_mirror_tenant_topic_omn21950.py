# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21950: the lane mirror reads its runtime's bus and writes the tenant topic.

Two runtimes, each with its own in-memory bus (the default the container
registers for ``ProtocolEventBus``), each hosting its own forwarder. Both
forwarders mirror the same contract-declared command topic onto one shared
parent bus. Before this change the mirror republished under the canonical name,
so both writers' records landed on one topic and the reader could not tell them
apart. Now a ``tenant_topics`` member is published under the runtime tenant's
physical topic, resolved by ``resolve_physical_topic``, and each tenant's
ingress receives only its own writer's request.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.idempotency import StoreIdempotencyInmemory
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayContainerSourceBus,
    ModelGatewayForwarderConfig,
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayLaneMirrorConfig,
    ModelGatewayPublishReceipt,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    resolve_physical_topic,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
    NodeLaneMirror,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror_bus_source import (
    IN_PROCESS_PARTITION,
    NodeLaneMirrorBusSource,
    build_lane_mirror_source,
    resolve_lane_mirror_source_bus,
)
from omnibase_infra.runtime.gateway_forwarder import build_gateway_transports

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = "onex.cmd.omnibase-infra.ledger-append.v1"
_PLAIN_TOPIC = "onex.evt.omnibase-infra.lane-heartbeat.v1"


class _ParentBusPublisher:
    """``ProtocolGatewayPublisher`` onto the shared parent bus (a test seam)."""

    def __init__(self, bus: EventBusInmemory) -> None:
        self._bus = bus

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> ModelGatewayPublishReceipt | None:
        raw = cast("dict[str, bytes]", headers or {})
        await self._bus.publish(
            topic,
            key,
            value,
            ModelEventHeaders(
                timestamp=datetime.now(UTC),
                source="lane-mirror",
                event_type="mirrored",
                message_id=UUID(raw["message_id"].decode()),
            ),
        )
        return None


def _identity(slug: str) -> ModelGatewayTenantIdentity:
    tenant_id = uuid4()
    return ModelGatewayTenantIdentity(
        tenant_id=tenant_id,
        tenant_slug=slug,
        principal_id=f"t-{tenant_id.hex}",
    )


def _lane_mirror(*, tenant_topics: tuple[str, ...]) -> ModelGatewayLaneMirrorConfig:
    return ModelGatewayLaneMirrorConfig(
        source_lane="runtime",
        mirror_lanes=("parent",),
        declared_lanes=("runtime", "parent"),
        topics=(_COMMAND_TOPIC, _PLAIN_TOPIC),
        tenant_topics=tenant_topics,
        poll_timeout_ms=50,
    )


def _runtime_config(slug: str) -> ModelGatewayForwarderRuntimeConfig:
    return ModelGatewayForwarderRuntimeConfig(
        forwarder=ModelGatewayForwarderConfig(
            tenant_identity=_identity(slug),
            local_transport_flavor="containerized",
            dedupe_store_path=Path(f"/app/data/{slug}.sqlite3"),
            lane_mirror=_lane_mirror(tenant_topics=(_COMMAND_TOPIC,)),
        ),
        lane_mirror_source_bus=ModelGatewayContainerSourceBus(source="container"),
        lane_mirror_buses={
            "parent": ModelKafkaEventBusConfig(
                bootstrap_servers="parent-redpanda:9092",
                environment="lane-mirror-parent",
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
        },
    )


class _Container:
    """The one container call the forwarder makes: resolve ``ProtocolEventBus``."""

    def __init__(self, bus: EventBusInmemory) -> None:
        self._bus = bus

    async def get_service_async(self, protocol_type: type[object]) -> object:
        assert protocol_type is ProtocolEventBus
        return self._bus


async def _start_runtime(
    slug: str, parent: EventBusInmemory
) -> tuple[EventBusInmemory, NodeLaneMirror, NodeLaneMirrorBusSource]:
    """One runtime: its own bus, from its container, and its own forwarder."""
    runtime_bus = EventBusInmemory(environment=slug, group="runtime")
    await runtime_bus.start()
    config = _runtime_config(slug)
    bus = await resolve_lane_mirror_source_bus(
        cast("ModelONEXContainer", _Container(runtime_bus))
    )
    source = build_lane_mirror_source(config, {}, bus)
    assert isinstance(source, NodeLaneMirrorBusSource)
    await source.start()
    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    mirror = NodeLaneMirror(
        config=lane_mirror,
        source_consumer=source,
        mirror_producers={"parent": _ParentBusPublisher(parent)},
        idempotency_store=StoreIdempotencyInmemory(),
        tenant_slug=config.forwarder.tenant_identity.tenant_slug,
    )
    return runtime_bus, mirror, source


async def _write(bus: EventBusInmemory, topic: str, value: bytes) -> None:
    await bus.publish(
        topic,
        None,
        value,
        ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="writer",
            event_type="command",
            message_id=uuid4(),
        ),
    )


@pytest.mark.asyncio
async def test_tenant_topic_two_runtimes_land_on_two_ingress_topics() -> None:
    parent = EventBusInmemory(environment="parent", group="reducer")
    await parent.start()
    received: dict[str, list[bytes]] = {"runtime-a": [], "runtime-b": []}

    def _collector(slug: str) -> Callable[[ModelEventMessage], Awaitable[None]]:
        async def _on(message: ModelEventMessage) -> None:
            received[slug].append(message.value)

        return _on

    for slug in received:
        await parent.subscribe(
            resolve_physical_topic(_COMMAND_TOPIC, tenant_slug=slug),
            on_message=_collector(slug),
            group_id=f"ingress-{slug}",
        )
    canonical: list[bytes] = []

    async def _on_canonical(message: ModelEventMessage) -> None:
        canonical.append(message.value)

    await parent.subscribe(
        _COMMAND_TOPIC, on_message=_on_canonical, group_id="canonical"
    )

    bus_a, mirror_a, source_a = await _start_runtime("runtime-a", parent)
    bus_b, mirror_b, source_b = await _start_runtime("runtime-b", parent)
    await _write(bus_a, _COMMAND_TOPIC, b"request-from-a")
    await _write(bus_b, _COMMAND_TOPIC, b"request-from-b")

    assert await mirror_a.drain_once() == 1
    assert await mirror_b.drain_once() == 1

    assert received == {
        "runtime-a": [b"request-from-a"],
        "runtime-b": [b"request-from-b"],
    }
    assert canonical == []
    assert mirror_a.destination_topic(_COMMAND_TOPIC) != mirror_b.destination_topic(
        _COMMAND_TOPIC
    )
    for source in (source_a, source_b):
        await source.close()


@pytest.mark.asyncio
async def test_tenant_topic_mapping_leaves_undeclared_topics_canonical() -> None:
    parent = EventBusInmemory(environment="parent", group="reducer")
    await parent.start()
    plain: list[bytes] = []

    async def _on(message: ModelEventMessage) -> None:
        plain.append(message.value)

    await parent.subscribe(_PLAIN_TOPIC, on_message=_on, group_id="plain")
    bus_a, mirror_a, source_a = await _start_runtime("runtime-a", parent)
    await _write(bus_a, _PLAIN_TOPIC, b"heartbeat")
    assert await mirror_a.drain_once() == 1
    assert plain == [b"heartbeat"]
    assert mirror_a.destination_topic(_PLAIN_TOPIC) == _PLAIN_TOPIC
    await source_a.close()


def test_tenant_topic_requires_the_tenant_slug() -> None:
    with pytest.raises(ValueError, match="tenant slug"):
        NodeLaneMirror(
            config=_lane_mirror(tenant_topics=(_COMMAND_TOPIC,)),
            source_consumer=cast("NodeLaneMirrorBusSource", object()),
            mirror_producers={"parent": cast("_ParentBusPublisher", object())},
            idempotency_store=StoreIdempotencyInmemory(),
        )


def test_tenant_topic_must_be_a_declared_topic() -> None:
    with pytest.raises(ValueError, match="must be declared lane_mirror topics"):
        ModelGatewayLaneMirrorConfig(
            source_lane="runtime",
            mirror_lanes=("parent",),
            declared_lanes=("runtime", "parent"),
            topics=(_PLAIN_TOPIC,),
            tenant_topics=(_COMMAND_TOPIC,),
        )


def test_tenant_topic_produce_binding_is_the_tenant_physical_topic() -> None:
    """The resolved ACL bindings name the tenant topic, and no source leg."""
    bindings, transports = build_gateway_transports(_runtime_config("runtime-a"))
    produced = {
        binding.physical_topic
        for binding in bindings.bindings
        if binding.broker == "lane.parent.kafka.broker"
    }
    assert produced == {
        resolve_physical_topic(_COMMAND_TOPIC, tenant_slug="runtime-a"),
        _PLAIN_TOPIC,
    }
    assert not any(
        binding.broker.startswith("lane.runtime") for binding in bindings.bindings
    )
    assert "lane_source" not in transports


@pytest.mark.asyncio
async def test_in_memory_source_records_carry_identity_and_in_process_partition() -> (
    None
):
    bus = EventBusInmemory(environment="runtime-a", group="runtime")
    await bus.start()
    source = build_lane_mirror_source(
        _runtime_config("runtime-a"),
        {},
        source_bus=cast("ProtocolEventBus", bus),
    )
    assert isinstance(source, NodeLaneMirrorBusSource)
    await source.start()
    message_id = uuid4()
    await bus.publish(
        _COMMAND_TOPIC,
        b"k",
        b"v",
        ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="writer",
            event_type="command",
            message_id=message_id,
        ),
    )
    (record,) = await source.poll(max_messages=10, timeout_ms=50)
    assert record.partition == IN_PROCESS_PARTITION
    assert record.headers["message_id"] == str(message_id).encode()
    await source.nack(record)
    (again,) = await source.poll(max_messages=10, timeout_ms=50)
    assert again == record
    assert await source.poll(max_messages=10, timeout_ms=10) == ()
    await source.close()


def test_source_bus_container_is_required_when_the_overlay_selects_it() -> None:
    with pytest.raises(ValueError, match="ProtocolEventBus"):
        build_lane_mirror_source(_runtime_config("runtime-a"), {}, None)


def test_source_bus_container_is_refused_for_a_kafka_selected_source() -> None:
    config = _runtime_config("runtime-a").model_copy(
        update={
            "lane_mirror_source_bus": ModelKafkaEventBusConfig(
                bootstrap_servers="runtime-redpanda:9092",
                environment="lane-mirror-source",
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
        }
    )
    with pytest.raises(ValueError, match="must not be handed in"):
        build_lane_mirror_source(
            config,
            {},
            source_bus=cast(
                "ProtocolEventBus", EventBusInmemory(environment="x", group="y")
            ),
        )
