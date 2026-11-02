# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lane-mirror source leg read from the hosting runtime's own bus (OMN-21950).

Every runtime has a bus: the in-memory bus by default, a hosted bus where its
overlay says so. The forwarder runs inside the runtime it forwards for, so when
the overlay selects ``lane_mirror_source_bus: {source: container}`` the source
leg is the ``ProtocolEventBus`` the runtime's container resolved -- never an
adapter this node constructs.

``NodeLaneMirror`` pulls (``poll`` / ``commit`` / ``nack``, the existing
``ProtocolGatewayConsumer`` surface) while ``ProtocolEventBus`` pushes
(``subscribe(on_message=...)``). This class is the seam between the two and
nothing more: it subscribes once per contract-declared topic, buffers each
delivered record as a ``ModelTransportMessage`` and hands it to the mirror. A
nacked record goes back to the head of the buffer so it is redelivered, which
is the retain-and-retry half of the mirror's ordering contract.

Named ``Node*`` for the reason its sibling ``NodeLaneMirror`` is: ``Service*``
is a hard-fail shape under the OMN-14350 lifecycle-class ratchet.

Coordinates. The mirror's loop guard (OMN-17201) treats a source coordinate
equal to a destination coordinate as proof the destination broker IS the
source broker. A record read from the runtime's bus is given partition
``IN_PROCESS_PARTITION`` (-1), a partition no broker assigns, so a coordinate
from this source can never equal a broker receipt by coincidence. An in-process
bus is never a mirror lane's broker, so the guard has nothing to detect here.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from itertools import count
from typing import cast

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
from omnibase_core.protocols.event_bus.protocol_event_message import (
    ProtocolEventMessage,
)
from omnibase_core.protocols.event_bus.protocol_node_identity import (
    ProtocolNodeIdentity,
)
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_runtime_config import (
    ModelGatewayForwarderRuntimeConfig,
)

IN_PROCESS_PARTITION = -1
"""Partition stamped on every record read from the runtime's own bus."""


def _encode_headers(headers: object) -> dict[str, bytes]:
    """Render bus headers as the raw ``str -> bytes`` map a transport carries.

    The same rendering ``TransportGatewayBus`` applies to ``ModelEventHeaders``
    on publish, so the mandatory ``message_id`` header the mirror keys its
    durable marker on crosses unchanged.
    """
    dump = getattr(headers, "model_dump", None)
    if not callable(dump):
        return {}
    fields = cast("Mapping[str, object]", dump(mode="json", exclude_none=True))
    return {name: str(value).encode("utf-8") for name, value in fields.items()}


class NodeLaneMirrorBusSource:
    """``ProtocolGatewayConsumer`` over a container-resolved ``ProtocolEventBus``.

    The buffer is created by ``start``; nothing is constructed in ``__init__``.
    """

    _buffer: deque[ModelTransportMessage]
    _available: asyncio.Event
    _offsets: Iterator[int]
    _unsubscribes: list[Callable[[], Awaitable[None]]]

    def __init__(
        self,
        *,
        event_bus: ProtocolEventBus,
        topics: Sequence[str],
        node_identity: ProtocolNodeIdentity,
    ) -> None:
        if not topics:
            raise ValueError("a lane-mirror bus source requires at least one topic")
        self._bus = event_bus
        self._topics = topics
        self._identity = node_identity
        self._started = False

    async def start(self) -> None:
        """Subscribe to every declared topic on the runtime's bus."""
        if self._started:
            return
        self._buffer = deque()
        self._available = asyncio.Event()
        self._offsets = count()
        self._unsubscribes = []
        self._started = True
        for topic in self._topics:
            self._unsubscribes.append(
                await self._bus.subscribe(topic, self._identity, self._receive)
            )

    async def close(self) -> None:
        """Drop every subscription; buffered records are not committed."""
        if not self._started:
            return
        unsubscribes, self._unsubscribes = self._unsubscribes, []
        for unsubscribe in unsubscribes:
            await unsubscribe()

    async def _receive(self, message: ProtocolEventMessage) -> None:
        self._buffer.append(
            ModelTransportMessage(
                topic=message.topic,
                partition=IN_PROCESS_PARTITION,
                offset=next(self._offsets),
                key=message.key,
                value=message.value,
                headers=_encode_headers(message.headers),
                ack_token=None,
            )
        )
        self._available.set()

    async def poll(
        self,
        *,
        max_messages: int,
        timeout_ms: int,
    ) -> Sequence[ModelTransportMessage]:
        """Return up to ``max_messages`` buffered records, waiting up to the timeout."""
        if not self._buffer:
            self._available.clear()
            try:
                await asyncio.wait_for(self._available.wait(), timeout_ms / 1000)
            except TimeoutError:
                return ()
        batch: list[ModelTransportMessage] = []
        while self._buffer and len(batch) < max_messages:
            batch.append(self._buffer.popleft())
        return tuple(batch)

    async def commit(self, message: object) -> None:
        """Nothing to persist: a polled record is already out of the buffer."""
        if not isinstance(message, ModelTransportMessage):
            raise TypeError("a lane-mirror bus source commits only its own records")

    async def nack(self, message: object) -> None:
        """Return the record to the head of the buffer for redelivery."""
        if not isinstance(message, ModelTransportMessage):
            raise TypeError("a lane-mirror bus source nacks only its own records")
        self._buffer.appendleft(message)
        self._available.set()


async def resolve_lane_mirror_source_bus(
    container: ModelONEXContainer,
) -> ProtocolEventBus:
    """The hosting runtime's bus, as its overlay registered it.

    The forwarder asks the container for ``ProtocolEventBus`` like every other
    handler: the in-memory bus by default, a hosted bus where the overlay
    registers one. Nothing here names a concrete bus class.
    """
    # The Protocol is the registry key; the container never instantiates it.
    key: object = ProtocolEventBus
    bus = await container.get_service_async(cast("type[object]", key))
    return cast("ProtocolEventBus", bus)


def build_lane_mirror_source(
    config: ModelGatewayForwarderRuntimeConfig,
    transports: Mapping[str, KafkaTransport],
    source_bus: ProtocolEventBus | None,
) -> KafkaTransport | NodeLaneMirrorBusSource | None:
    """The lane-mirror source leg the overlay selected.

    A container-selected source reads the runtime's own ``ProtocolEventBus``,
    which the hosting runtime must hand in; a Kafka-selected source is the
    ``lane_source`` transport ``build_gateway_transports`` built. A bus handed
    in for a Kafka-selected source is refused rather than ignored, so the
    overlay stays the only thing that decides which bus is read.
    """
    lane_mirror = config.forwarder.lane_mirror
    if lane_mirror is None:
        if source_bus is not None:
            raise ValueError(
                "a source_bus was handed in but the node contract declares no "
                "lane_mirror to read it"
            )
        return None
    if not config.lane_mirror_source_from_container:
        if source_bus is not None:
            raise ValueError(
                "the overlay selects a Kafka lane_mirror_source_bus, so the "
                "runtime's container bus must not be handed in as the source"
            )
        return transports["lane_source"]
    if source_bus is None:
        raise ValueError(
            "the overlay selects lane_mirror_source_bus {source: container}, so "
            "the forwarder must run inside its runtime and be handed that "
            "runtime's ProtocolEventBus (resolve_lane_mirror_source_bus)"
        )
    identity = config.forwarder.tenant_identity
    return NodeLaneMirrorBusSource(
        event_bus=source_bus,
        topics=lane_mirror.topics,
        node_identity=ModelNodeIdentity(
            env=identity.tenant_slug,
            service="omnibase_infra",
            node_name="node_bus_forwarder_effect",
            version="v1",
        ),
    )


__all__ = [
    "IN_PROCESS_PARTITION",
    "NodeLaneMirrorBusSource",
    "build_lane_mirror_source",
    "resolve_lane_mirror_source_bus",
]
