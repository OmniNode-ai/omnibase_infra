# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Claim/publish/ack seam over the local mirrored delegation topics.

``ProtocolDelegationChannel`` is the one seam the worker cycle depends on --
everything in ``worker_cycle.py`` is written against this Protocol, not
against aiokafka directly, so the cycle logic is unit-testable with an
in-memory fake (see the test suite) without a running broker.

``AiokafkaDelegationChannel`` is the real implementation, consuming the
inbound mirror topics (``DELEGATION_REQUEST_TOPIC`` /
``DELEGATION_INFERENCE_REQUEST_TOPIC``) and producing to the outbound
mirror topics declared in ``node_bus_forwarder_effect``'s contract. It talks
only to the LOCAL broker the forwarder mirrors onto -- never to the cloud
Kafka edge directly, matching the "cloud never sees LAN" split (the
forwarder's own outbound leg is what reaches the cloud edge, and it
initiates that connection itself).

Both the consumer and the producer are constructor-injected rather than
built inside this class, so tests can supply fakes implementing the same
narrow ``Protocol*`` surface aiokafka exposes (``start``/``stop``/
``getone``/``commit`` and ``start``/``stop``/``send_and_wait``) without
depending on aiokafka's wire protocol or a live broker. ``build_kafka_channel``
is the one place that constructs the real aiokafka objects; it is a thin
factory, not exercised by the unit suite (it would require a live broker).
"""

from __future__ import annotations

import json
import logging
from typing import Protocol
from uuid import UUID

from omnibase_core.types import JsonType
from scripts.edge_delegation_worker.models import ModelDelegationEnvelope
from scripts.edge_delegation_worker.topic_constants import INBOUND_TOPICS

logger = logging.getLogger(__name__)


class ProtocolDelegationChannel(Protocol):
    """The claim/publish/ack seam the worker cycle is written against."""

    async def claim(self) -> ModelDelegationEnvelope | None:
        """Return one unclaimed envelope, or ``None`` if none is available now."""
        ...

    async def publish_result(
        self,
        *,
        topic: str,
        correlation_id: UUID,
        event_type: str,
        payload: dict[str, JsonType],
    ) -> None:
        """Publish one result envelope onto *topic* for the forwarder to mirror out."""
        ...

    async def ack(self, envelope: ModelDelegationEnvelope) -> None:
        """Commit the claimed envelope's offset -- it will not be redelivered."""
        ...

    async def nack(self, envelope: ModelDelegationEnvelope, *, reason: str) -> None:
        """Leave the envelope's offset uncommitted so it is redelivered."""
        ...


class _KafkaMessageLike(Protocol):
    value: bytes
    topic: str
    headers: list[tuple[str, bytes]]


class ProtocolAsyncKafkaConsumer(Protocol):
    """The narrow slice of ``aiokafka.AIOKafkaConsumer`` this module needs."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def getone(self) -> _KafkaMessageLike: ...
    async def commit(self) -> None: ...


class ProtocolAsyncKafkaProducer(Protocol):
    """The narrow slice of ``aiokafka.AIOKafkaProducer`` this module needs."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        *,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object: ...


def _decode_envelope(message: _KafkaMessageLike) -> ModelDelegationEnvelope | None:
    """Decode one raw Kafka message into a typed envelope, or ``None`` on malformed input.

    Fail-closed by design: a malformed message is never guessed into a
    usable envelope. The caller (``claim``) treats ``None`` as "skip and
    keep polling," never as "empty queue."
    """
    try:
        decoded = json.loads(message.value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning(
            "edge_delegation_worker: unparseable message on topic=%s, skipping",
            message.topic,
        )
        return None
    if not isinstance(decoded, dict):
        logger.warning(
            "edge_delegation_worker: non-object message body on topic=%s, skipping",
            message.topic,
        )
        return None
    try:
        return ModelDelegationEnvelope(
            correlation_id=UUID(str(decoded["correlation_id"])),
            source_topic=message.topic,
            event_type=str(decoded.get("event_type", "")),
            payload=decoded.get("payload", {}),
            headers={k: v.decode("utf-8", "replace") for k, v in message.headers},
        )
    except (KeyError, ValueError) as exc:
        logger.warning(
            "edge_delegation_worker: envelope shape mismatch on topic=%s: %s",
            message.topic,
            exc,
        )
        return None


class AiokafkaDelegationChannel:
    """Local-bus-backed implementation of ``ProtocolDelegationChannel``."""

    def __init__(
        self,
        *,
        consumer: ProtocolAsyncKafkaConsumer,
        producer: ProtocolAsyncKafkaProducer,
    ) -> None:
        self._consumer = consumer
        self._producer = producer
        self._started = False

    async def start(self) -> None:
        await self._consumer.start()
        await self._producer.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._producer.stop()
        await self._consumer.stop()
        self._started = False

    async def claim(self) -> ModelDelegationEnvelope | None:
        message = await self._consumer.getone()
        return _decode_envelope(message)

    async def publish_result(
        self,
        *,
        topic: str,
        correlation_id: UUID,
        event_type: str,
        payload: dict[str, JsonType],
    ) -> None:
        body = {
            "correlation_id": str(correlation_id),
            "event_type": event_type,
            "payload": payload,
        }
        await self._producer.send_and_wait(topic, json.dumps(body).encode("utf-8"))

    async def ack(self, envelope: ModelDelegationEnvelope) -> None:
        await self._consumer.commit()

    async def nack(self, envelope: ModelDelegationEnvelope, *, reason: str) -> None:
        logger.warning(
            "edge_delegation_worker: nack correlation_id=%s topic=%s reason=%s "
            "(offset left uncommitted, will be redelivered)",
            envelope.correlation_id,
            envelope.source_topic,
            reason,
        )


def build_kafka_channel(
    *,
    brokers: str,
    consumer_group: str,
) -> AiokafkaDelegationChannel:
    """Construct the real, live-broker-backed channel.

    Not covered by the unit suite -- it requires a real Kafka/Redpanda
    broker to start against, which is exactly the "DO NOT attach to any
    live queue" boundary this build stays inside of. Import is deferred to
    keep aiokafka's import cost out of every module that only needs the
    ``ProtocolDelegationChannel`` seam (the fake-backed unit tests included).
    """
    from aiokafka import (
        AIOKafkaConsumer,
        AIOKafkaProducer,
    )  # local import: see docstring

    consumer = AIOKafkaConsumer(
        *INBOUND_TOPICS,
        bootstrap_servers=brokers,
        group_id=consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="latest",
    )
    producer = AIOKafkaProducer(bootstrap_servers=brokers)
    return AiokafkaDelegationChannel(consumer=consumer, producer=producer)
