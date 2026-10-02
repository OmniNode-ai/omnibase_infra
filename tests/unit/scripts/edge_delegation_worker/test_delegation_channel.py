# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts.edge_delegation_worker.delegation_channel.

Uses small fakes that satisfy ``ProtocolAsyncKafkaConsumer`` /
``ProtocolAsyncKafkaProducer`` structurally -- no real broker, no aiokafka
import at collection time for the class-under-test itself (``ack`` and
``build_kafka_channel`` defer their aiokafka imports).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

from scripts.edge_delegation_worker.delegation_channel import AiokafkaDelegationChannel
from scripts.edge_delegation_worker.models import ModelDelegationEnvelope
from scripts.edge_delegation_worker.topic_constants import DELEGATION_COMPLETED_TOPIC

pytestmark = pytest.mark.unit


@dataclass
class _FakeMessage:
    value: bytes
    topic: str
    partition: int = 0
    offset: int = 0
    headers: list[tuple[str, bytes]] = field(default_factory=list)


class _FakeConsumer:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self._messages = list(messages)
        self.started = False
        self.stopped = False
        self.commits: list[Mapping[TopicPartition, int]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getone(self) -> _FakeMessage:
        return self._messages.pop(0)

    async def commit(self, offsets: Mapping[TopicPartition, int]) -> None:
        self.commits.append(dict(offsets))


class _FakeProducer:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.sent: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        *,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object:
        self.sent.append((topic, value))
        return None


def _envelope_message(
    correlation_id: str,
    *,
    topic: str = "onex.cmd.x",
    partition: int = 0,
    offset: int = 0,
) -> _FakeMessage:
    body = {
        "correlation_id": correlation_id,
        "event_type": "omnibase-infra.delegation-request",
        "payload": {"prompt": "hello"},
    }
    return _FakeMessage(
        value=json.dumps(body).encode("utf-8"),
        topic=topic,
        partition=partition,
        offset=offset,
    )


@pytest.mark.asyncio
async def test_claim_decodes_a_valid_envelope() -> None:
    correlation_id = str(uuid4())
    consumer = _FakeConsumer([_envelope_message(correlation_id)])
    producer = _FakeProducer()
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=producer)

    envelope = await channel.claim()
    assert envelope is not None
    assert str(envelope.correlation_id) == correlation_id
    assert envelope.payload == {"prompt": "hello"}


@pytest.mark.asyncio
async def test_claim_carries_messages_partition_and_offset() -> None:
    message = _envelope_message(str(uuid4()), partition=3, offset=262)
    consumer = _FakeConsumer([message])
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=_FakeProducer())

    envelope = await channel.claim()

    assert envelope is not None
    assert envelope.source_topic == message.topic
    assert envelope.source_partition == message.partition
    assert envelope.source_offset == message.offset


@pytest.mark.asyncio
async def test_claim_returns_none_for_malformed_json() -> None:
    consumer = _FakeConsumer([_FakeMessage(value=b"not json", topic="onex.cmd.x")])
    producer = _FakeProducer()
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=producer)

    assert await channel.claim() is None


@pytest.mark.asyncio
async def test_claim_returns_none_for_missing_correlation_id() -> None:
    body = json.dumps({"event_type": "x", "payload": {}}).encode("utf-8")
    consumer = _FakeConsumer([_FakeMessage(value=body, topic="onex.cmd.x")])
    producer = _FakeProducer()
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=producer)

    assert await channel.claim() is None


@pytest.mark.asyncio
async def test_ack_commits_offset() -> None:
    consumer = _FakeConsumer([])
    producer = _FakeProducer()
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=producer)

    envelope = ModelDelegationEnvelope(
        correlation_id=uuid4(),
        source_topic="onex.cmd.x",
        source_partition=0,
        source_offset=0,
        event_type="e",
        payload={},
    )
    await channel.ack(envelope)
    assert consumer.commits == [{TopicPartition("onex.cmd.x", 0): 1}]


@pytest.mark.asyncio
async def test_ack_commits_only_the_claimed_records_partition_offset() -> None:
    topic = "onex.cmd.x"
    claimed = _envelope_message(str(uuid4()), topic=topic, partition=3, offset=262)
    buffered = _envelope_message(str(uuid4()), topic=topic, partition=3, offset=263)
    consumer = _FakeConsumer([claimed, buffered])
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=_FakeProducer())

    envelope = await channel.claim()
    assert envelope is not None
    await channel.ack(envelope)

    assert consumer.commits == [{TopicPartition(topic, 3): 263}]
    assert consumer._messages == [buffered]


@pytest.mark.asyncio
async def test_nack_does_not_commit_offset() -> None:
    consumer = _FakeConsumer([])
    producer = _FakeProducer()
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=producer)

    envelope = ModelDelegationEnvelope(
        correlation_id=uuid4(),
        source_topic="onex.cmd.x",
        source_partition=0,
        source_offset=0,
        event_type="e",
        payload={},
    )
    await channel.nack(envelope, reason="local model unavailable")
    assert consumer.commits == []


@pytest.mark.asyncio
async def test_publish_result_sends_typed_envelope() -> None:
    consumer = _FakeConsumer([])
    producer = _FakeProducer()
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=producer)

    correlation_id = uuid4()
    await channel.publish_result(
        topic=DELEGATION_COMPLETED_TOPIC,
        correlation_id=correlation_id,
        event_type="omnibase-infra.delegation-completed",
        payload={"content": "hi"},
    )

    assert len(producer.sent) == 1
    topic, raw_value = producer.sent[0]
    assert topic == DELEGATION_COMPLETED_TOPIC
    decoded = json.loads(raw_value.decode("utf-8"))
    assert decoded["correlation_id"] == str(correlation_id)
    assert decoded["payload"] == {"content": "hi"}


@pytest.mark.asyncio
async def test_start_stop_lifecycle() -> None:
    consumer = _FakeConsumer([])
    producer = _FakeProducer()
    channel = AiokafkaDelegationChannel(consumer=consumer, producer=producer)

    await channel.start()
    assert consumer.started
    assert producer.started

    await channel.stop()
    assert consumer.stopped
    assert producer.stopped
