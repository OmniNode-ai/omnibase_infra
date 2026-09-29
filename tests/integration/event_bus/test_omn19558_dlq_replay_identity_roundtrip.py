# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dead-lettered command keeps its consumer identity when replayed."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.integration

# onex-topic-allow: the delegate-skill command topic whose replay identity matters.
_COMMAND_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
# onex-topic-allow: the in-process DLQ publication destination.
_DLQ_TOPIC = "onex.dlq.omnibase-infra.intents.v1"


class _CapturingProducer:
    """Capture the exact Kafka wire arguments without opening a connection."""

    def __init__(self) -> None:
        self.records: list[
            tuple[str, bytes, bytes | None, list[tuple[str, bytes | None]]]
        ] = []

    async def send_and_wait(
        self,
        topic: str,
        *,
        value: bytes,
        key: bytes | None,
        headers: list[tuple[str, bytes]],
    ) -> object:
        self.records.append(
            (topic, value, key, [(name, data) for name, data in headers])
        )
        return object()


@pytest.mark.asyncio
async def test_dead_letter_replay_keeps_the_original_consumer_message_id() -> None:
    """Exercise the DLQ writer, replay reader, publisher and consumer parser."""
    original_id = uuid4()
    correlation_id = uuid4()
    body = json.dumps(
        {"envelope_id": str(uuid4()), "payload": {"skill": "example"}}
    ).encode()
    failed = ModelEventMessage(
        topic=_COMMAND_TOPIC,
        key=b"delegate-1",
        value=body,
        headers=ModelEventHeaders(
            source="integration-test",
            event_type="delegate-skill",
            timestamp=datetime.now(UTC),
            message_id=original_id,
            correlation_id=correlation_id,
        ),
        offset="41",
        partition=0,
    )
    bus = EventBusKafka(
        ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            dead_letter_topic=_DLQ_TOPIC,
        )
    )
    dlq_wire = _CapturingProducer()
    bus._producer = dlq_wire
    assert await bus._publish_to_dlq(
        _COMMAND_TOPIC,
        failed,
        TimeoutError("dispatch deadline exceeded"),
        correlation_id,
        consumer_group="delegate-skill-integration",
    )
    assert len(dlq_wire.records) == 1
    dlq_topic, dlq_bytes, _, _ = dlq_wire.records[0]
    assert dlq_topic == _DLQ_TOPIC

    parsed = ModelDlqMessage.from_kafka_message(
        json.loads(dlq_bytes), dlq_offset=7, dlq_partition=0
    )
    assert parsed.original_message_id == original_id
    assert parsed.original_value.encode() == body

    replay = DLQProducer(
        ModelDlqReplayEngineConfig(
            bootstrap_servers="localhost:9092", dlq_topic=_DLQ_TOPIC
        )
    )
    replay_wire = _CapturingProducer()
    replay._producer = replay_wire
    replay._started = True
    await replay.replay_message(parsed, uuid4())
    assert len(replay_wire.records) == 1
    replay_topic, replay_bytes, replay_key, replay_headers = replay_wire.records[0]
    assert replay_topic == _COMMAND_TOPIC
    assert replay_bytes == body
    assert replay_key == failed.key
    assert bus._kafka_headers_to_model(replay_headers).message_id == original_id
