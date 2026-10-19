# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise no-dispatcher recovery through the runtime and replay reader.

The real dispatch engine, wiring callback, Kafka consume loop and DLQ writer
run together. Only the Kafka transport is replaced: the consumer models fetch
positions and redelivery, and the producer persists accepted bytes to disk.
This is a hosted-runner runtime probe, not evidence of a deployed rollout or
of Kafka's background auto-commit scheduling.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

import pytest
from aiokafka.errors import UnknownTopicOrPartitionError
from aiokafka.structs import ConsumerRecord, TopicPartition

import omnibase_infra.event_bus.event_bus_kafka as kafka_module
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.event_bus import ModelDlqConfig
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

# onex-topic-allow: reproduce the command and category sink from OMN-14936.
_TOPIC = "onex.cmd.omnibase-infra.delegation-inference-request.v1"
_DLQ = "onex.dlq.omnibase-infra.commands.v1"
_GROUP = "dev.omn14936-recovery"
_OFFSET = 200


class _DurableTransport:
    """Refuse writes during the outage; fsync accepted category-DLQ bytes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writable = False
        self.accepted_topics: list[str] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_and_wait(self, topic: str, **kwargs: Any) -> object:
        if not self.writable or topic != _DLQ:
            raise UnknownTopicOrPartitionError("DLQ unavailable")
        with self.path.open("ab") as stream:
            stream.write(kwargs["value"] + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.accepted_topics.append(topic)
        return object()


class _RedeliveringConsumer:
    """Fetch advances the position; only seek makes the record visible again."""

    def __init__(
        self,
        record: ConsumerRecord,
        producer: _DurableTransport,
        blocked_polls: int,
        stop: Callable[[], None],
    ) -> None:
        self.record = record
        self.producer = producer
        self.blocked_polls = blocked_polls
        self.stop_loop = stop
        self.position = _OFFSET
        self.polls = 0
        self.positions_before_poll: list[int] = []
        self.seeks: list[tuple[TopicPartition, int]] = []

    async def getmany(
        self, *args: object, **kwargs: object
    ) -> dict[TopicPartition, list[ConsumerRecord]]:
        self.positions_before_poll.append(self.position)
        self.polls += 1
        self.producer.writable = self.polls > self.blocked_polls
        if self.position > _OFFSET or self.polls > self.blocked_polls + 2:
            self.stop_loop()
            return {}
        self.position = _OFFSET + 1
        return {TopicPartition(_TOPIC, 0): [self.record]}

    def assignment(self) -> set[TopicPartition]:
        return set()

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seeks.append((partition, offset))
        self.position = offset

    async def stop(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_polls", [1, 2])
async def test_no_dispatcher_rewinds_until_dlq_recovers_then_is_replayable(
    tmp_path: Path, blocked_polls: int
) -> None:
    """No durable copy means redelivery; recovery leaves exactly one copy."""
    correlation_id = uuid4()
    payload = {
        "event_type": "delegation-inference-request",
        "correlation_id": str(correlation_id),
        "payload": {"prompt": "omn14936-recovery"},
    }
    value = json.dumps(payload).encode()
    record = ConsumerRecord(
        topic=_TOPIC,
        partition=0,
        offset=_OFFSET,
        timestamp=0,
        timestamp_type=0,
        key=b"omn14936",
        value=value,
        checksum=None,
        serialized_key_size=8,
        serialized_value_size=len(value),
        headers=[],
    )
    producer = _DurableTransport(tmp_path / "dlq.jsonl")
    engine = MessageDispatchEngine()
    engine.freeze()
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092", environment="dev"
    )
    assert config.enable_auto_commit is True
    with (
        patch.object(kafka_module, "AIOKafkaProducer", return_value=producer),
        patch.object(kafka_module, "DLQ_UNPERSISTED_REWIND_BACKOFF_SECONDS", 0.0),
    ):
        bus = EventBusKafka(config=config)
        await bus.start()
        try:
            wiring = EventBusSubcontractWiring(
                event_bus=cast("Any", bus),
                dispatch_engine=engine,
                environment="dev",
                node_name="omn14936-recovery",
                service="test-service",
                version="v1",
                dlq_config=ModelDlqConfig(
                    enabled=True, on_content_error="dlq_and_commit"
                ),
            )
            callback = wiring._create_dispatch_callback(_TOPIC, _GROUP)
            consumer = _RedeliveringConsumer(
                record, producer, blocked_polls, lambda: setattr(bus, "_shutdown", True)
            )
            bus._group_consumers[(_TOPIC, _GROUP)] = cast("Any", consumer)
            bus._subscribers[_TOPIC] = [(_GROUP, "recovery", cast("Any", callback))]
            await asyncio.wait_for(bus._consume_loop(_TOPIC, _GROUP, uuid4()), 10)
        finally:
            await bus.close()

    assert consumer.seeks == [(TopicPartition(_TOPIC, 0), _OFFSET)] * blocked_polls
    assert consumer.positions_before_poll == [_OFFSET] * (blocked_polls + 1) + [
        _OFFSET + 1
    ]
    assert consumer.position == _OFFSET + 1
    assert producer.accepted_topics == [_DLQ]
    # Reopen after closing the producer: assert persisted wire bytes, not calls.
    records = producer.path.read_bytes().splitlines()
    assert len(records) == 1
    envelope = json.loads(records[0])
    assert envelope["failure_class"] == "no_dispatcher"
    assert envelope["original_topic"] == _TOPIC
    replay = ModelDlqMessage.from_kafka_message(envelope, dlq_offset=0, dlq_partition=0)
    assert replay.original_topic == _TOPIC
    assert replay.correlation_id == correlation_id
    assert json.loads(replay.original_value) == payload
    assert replay.original_offset == str(_OFFSET)
