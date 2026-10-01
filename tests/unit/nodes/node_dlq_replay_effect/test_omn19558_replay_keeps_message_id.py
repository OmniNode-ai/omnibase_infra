# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19558 AC2 -- a DLQ-replayed record keeps its original message_id.

Measured 2026-09-25: node_dlq_replay_effect replayed a dead-lettered
delegate-skill command carrying only x-replay-* headers, so the consumer minted
a fresh message id and the OMN-18887 delivery claim treated every replay as a
new command. The replay must carry the id the record was first delivered under.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.unit

# onex-topic-allow: the live command topic the exercise measured.
_CMD_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
# onex-topic-allow: the DLQ topic this node drains.
_DLQ_TOPIC = "onex.dlq.omnibase-infra.intents.v1"
# onex-topic-allow: the durable quarantine sink.
_QUARANTINE_TOPIC = "onex.dlq.omnibase-infra.quarantine.v1"


class _FakeKafkaProducer:
    def __init__(self) -> None:
        self.headers: list[list[tuple[str, bytes]]] = []

    async def send_and_wait(self, topic: str, **kwargs: Any) -> None:
        self.headers.append(list(kwargs["headers"]))


def _payload(*, body: dict[str, object], original_message_id: UUID | None) -> dict:
    original_message: dict[str, object] = {
        "key": None,
        "value": json.dumps(body),
        "offset": 41,
        "partition": 0,
    }
    if original_message_id is not None:
        original_message["message_id"] = str(original_message_id)
    return {
        "original_topic": _CMD_TOPIC,
        "original_message": original_message,
        "failure_reason": "dispatch deadline exceeded",
        "failure_timestamp": "2026-09-25T13:02:22+00:00",
        "correlation_id": str(uuid4()),
        "retry_count": 0,
        "error_type": "dispatch_deadline_exceeded",
    }


async def _replay(message: ModelDlqMessage) -> dict[str, bytes]:
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_DLQ_TOPIC,
        quarantine_topic=_QUARANTINE_TOPIC,
    )
    producer = DLQProducer(config)
    fake = _FakeKafkaProducer()
    producer._producer = fake  # type: ignore[assignment]
    producer._started = True
    await producer.replay_message(message, uuid4())
    return dict(fake.headers[0])


@pytest.mark.asyncio
async def test_replay_carries_message_id_recorded_on_the_dlq_row() -> None:
    original = uuid4()
    message = ModelDlqMessage.from_kafka_message(
        _payload(body={"envelope_id": str(uuid4())}, original_message_id=original),
        dlq_offset=7,
        dlq_partition=0,
    )
    headers = await _replay(message)
    assert headers["message_id"] == str(original).encode()


@pytest.mark.asyncio
async def test_replay_of_a_row_without_recorded_id_falls_back_to_body_envelope_id() -> (
    None
):
    envelope_id = uuid4()
    message = ModelDlqMessage.from_kafka_message(
        _payload(body={"envelope_id": str(envelope_id)}, original_message_id=None),
        dlq_offset=8,
        dlq_partition=0,
    )
    headers = await _replay(message)
    assert headers["message_id"] == str(envelope_id).encode()


@pytest.mark.asyncio
async def test_replay_with_no_derivable_id_publishes_no_message_id_header() -> None:
    """Positive control: an id is never invented on the replay side."""
    message = ModelDlqMessage.from_kafka_message(
        _payload(body={"note": "no id here"}, original_message_id=None),
        dlq_offset=9,
        dlq_partition=0,
    )
    headers = await _replay(message)
    assert "message_id" not in headers
    assert "x-replay-count" in headers
