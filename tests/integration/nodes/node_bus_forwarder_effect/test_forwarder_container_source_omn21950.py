# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21950: a resolved overlay selecting the container bus drives a real mirror.

The overlay arrives as the plain mapping the runtime loads from YAML. The
mirror opens its source through ``NodeLaneMirror.open_source`` (the path
``run_gateway_forwarder`` takes), reads the runtime's in-memory bus, and
republishes a ``tenant_topics`` member under the runtime tenant's topic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.idempotency import StoreIdempotencyInmemory
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayPublishReceipt,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    resolve_physical_topic,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
    NodeLaneMirror,
)

pytestmark = pytest.mark.integration

_TOPIC = "onex.cmd.omnibase-infra.ledger-append.v1"
_SLUG = "runtime-a"


class _ParentPublisher:
    def __init__(self, bus: EventBusInmemory) -> None:
        self._bus = bus

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> ModelGatewayPublishReceipt | None:
        raw = headers if isinstance(headers, dict) else {}
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


def _overlay() -> dict[str, object]:
    tenant_id = uuid4()
    return {
        "forwarder": {
            "tenant_identity": {
                "tenant_id": str(tenant_id),
                "tenant_slug": _SLUG,
                "principal_id": f"t-{tenant_id.hex}",
            },
            "local_transport_flavor": "containerized",
            "dedupe_store_path": "/app/data/a.sqlite3",
            "lane_mirror": {
                "source_lane": "runtime",
                "mirror_lanes": ["parent"],
                "declared_lanes": ["runtime", "parent"],
                "topics": [_TOPIC],
                "tenant_topics": [_TOPIC],
                "poll_timeout_ms": 50,
            },
        },
        "lane_mirror_source_bus": {"source": "container"},
        "lane_mirror_buses": {
            "parent": {
                "bootstrap_servers": "parent-redpanda:9092",
                "environment": "lane-mirror-parent",
                "enable_auto_commit": False,
                "auto_offset_reset": "earliest",
            }
        },
    }


@pytest.mark.asyncio
async def test_yaml_shaped_container_overlay_mirrors_under_tenant_topic() -> None:
    config = ModelGatewayForwarderRuntimeConfig.model_validate(_overlay())
    assert config.lane_mirror_source_from_container is True
    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None

    runtime_bus = EventBusInmemory(environment=_SLUG, group="runtime")
    await runtime_bus.start()
    parent = EventBusInmemory(environment="parent", group="reducer")
    await parent.start()
    landed: list[bytes] = []

    def _collector() -> Callable[[ModelEventMessage], Awaitable[None]]:
        async def _on(message: ModelEventMessage) -> None:
            landed.append(message.value)

        return _on

    await parent.subscribe(
        resolve_physical_topic(_TOPIC, tenant_slug=_SLUG),
        on_message=_collector(),
        group_id="ingress",
    )

    source = NodeLaneMirror.open_source(
        config, {}, cast("ProtocolEventBus", runtime_bus)
    )
    await source.start()
    mirror = NodeLaneMirror(
        config=lane_mirror,
        source_consumer=source,
        mirror_producers={"parent": _ParentPublisher(parent)},
        idempotency_store=StoreIdempotencyInmemory(),
        tenant_slug=_SLUG,
    )
    await runtime_bus.publish(
        _TOPIC,
        None,
        b"request",
        ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="writer",
            event_type="command",
            message_id=uuid4(),
        ),
    )

    assert await mirror.drain_once() == 1
    assert landed == [b"request"]
    await source.close()
