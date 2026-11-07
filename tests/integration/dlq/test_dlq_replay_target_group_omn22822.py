# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22822 -- a replayed dead letter reaches only the group that failed it.

Drives the real envelope writer (``MixinKafkaDlq``), the real ``ModelDlqMessage``
parse, the real ``DLQProducer.replay_message`` (only the aiokafka producer is a
capture) and the real consume-side target read, over one source record that two
consumer groups rejected. No broker is needed and nothing is skipped.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.topic_constants import build_dlq_topic
from omnibase_infra.models.event_bus.model_dlq_replay_headers import (
    REPLAY_TARGET_GROUP_HEADER,
    ModelDlqReplayTarget,
)
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    DLQProducer,
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.topics import SUFFIX_PLATFORM_DLQ_MESSAGE

pytestmark = pytest.mark.integration

_DLQ_TOPIC = build_dlq_topic("events")
_GROUP_A = "dev.omnibase-infra.node_subscriber_a.consume.v1"
_GROUP_B = "dev.omnibase-infra.node_subscriber_b.consume.v1"
_BODY = json.dumps(
    {"event_type": "fixture", "payload": {"tick_id": "omn22822"}}
).encode()


class _Capture:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_and_wait(self, topic: str, **kwargs: Any) -> None:
        self.sent.append({"topic": topic, **kwargs})


class _Host(MixinKafkaDlq):
    def __init__(self) -> None:
        self._config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092", dead_letter_topic=_DLQ_TOPIC
        )
        self._environment = "test"
        self._group = "test-group"
        self._producer = cast("Any", _Capture())
        self._producer_lock = asyncio.Lock()
        self._timeout_seconds = 5
        self._init_dlq()

    def _model_headers_to_kafka(self, headers: object) -> list[tuple[str, bytes]]:
        return []


class _Raw:
    def __init__(self, headers: object = ()) -> None:
        self.key = b"k"
        self.value = _BODY
        self.offset = 7
        self.partition = 0
        self.headers = headers


async def _dead_letter(group: str, offset: int) -> ModelDlqMessage:
    host = _Host()
    assert await host._publish_raw_to_dlq(
        original_topic=SUFFIX_PLATFORM_DLQ_MESSAGE,
        raw_msg=_Raw(),
        error=RuntimeError("subscriber rejected the record"),
        correlation_id=UUID(int=22822),
        failure_type="handler_exception",
        consumer_group=group,
    )
    payload = json.loads(cast("_Capture", host._producer).sent[0]["value"])
    return ModelDlqMessage.from_kafka_message(
        payload, dlq_offset=offset, dlq_partition=0
    )


async def _replay(message: ModelDlqMessage) -> list[tuple[str, bytes]]:
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic=_DLQ_TOPIC,
        quarantine_topic=build_dlq_topic("quarantine"),
    )
    producer = DLQProducer(config)
    capture = _Capture()
    producer._producer = cast("Any", capture)
    producer._started = True
    await producer.replay_message(message, uuid4())
    return list(capture.sent[0]["headers"])


@pytest.mark.asyncio
async def test_golden_replay_reaches_only_the_group_that_failed_it() -> None:
    headers = await _replay(await _dead_letter(_GROUP_A, 0))
    assert dict(headers)[REPLAY_TARGET_GROUP_HEADER] == _GROUP_A.encode()

    target = ModelDlqReplayTarget.from_raw_msg(_Raw(headers))
    assert not target.excludes(_GROUP_A)
    assert target.excludes(_GROUP_B)


@pytest.mark.asyncio
async def test_error_chain_a_dead_letter_without_a_group_reaches_everyone() -> None:
    legacy = {
        "original_topic": SUFFIX_PLATFORM_DLQ_MESSAGE,
        "original_message": {"key": None, "value": _BODY.decode(), "offset": 1},
        "correlation_id": str(uuid4()),
    }
    message = ModelDlqMessage.from_kafka_message(legacy, dlq_offset=0, dlq_partition=0)
    headers = await _replay(message)
    assert REPLAY_TARGET_GROUP_HEADER not in dict(headers)

    target = ModelDlqReplayTarget.from_raw_msg(_Raw(headers))
    assert not target.excludes(_GROUP_A)
    assert not target.excludes(_GROUP_B)
