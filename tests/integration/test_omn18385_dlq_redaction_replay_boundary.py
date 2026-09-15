# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18385 integration proof for the DLQ redaction/replay boundary."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.utils.util_dlq_credential_redaction import DLQ_REDACTION_MARKER

pytestmark = pytest.mark.integration

_DLQ_TOPIC = "onex.dlq.omnibase-infra.commands.v1"  # onex-topic-allow
_ORIGINAL_TOPIC = (
    "onex.cmd.omnibase-infra.gateway-attach-request.v1"  # onex-topic-allow
)
_SENTINEL_TOKEN = "OMN18385-INTEGRATION-SENTINEL-NOT-A-SECRET"
_SENTINEL_EDGE = "edge-integration-proof"


class _CapturingProducer:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_and_wait(
        self,
        topic: str,
        *,
        value: bytes,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> object:
        self.sent.append(
            {"topic": topic, "value": value, "key": key, "headers": headers}
        )
        return object()


class _DlqHost(MixinKafkaDlq):
    def __init__(self) -> None:
        self._config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            dead_letter_topic=_DLQ_TOPIC,
        )
        self._environment = "test"
        self._group = "test-group"
        self._producer = _CapturingProducer()  # type: ignore[assignment]
        self._producer_lock = asyncio.Lock()
        self._timeout_seconds = 5
        self._init_dlq()

    def _model_headers_to_kafka(self, headers: object) -> list[tuple[str, bytes]]:
        return []


class _RawRecord:
    def __init__(self, value: bytes) -> None:
        self.key = b"edge-integration-proof"
        self.value = value
        self.offset = 137
        self.partition = 0
        self.headers = ()


async def test_redacted_raw_dlq_envelope_is_not_replay_eligible() -> None:
    host = _DlqHost()
    body = json.dumps(
        {
            "event_type": "gateway.attach",
            "payload": {
                "access_token": _SENTINEL_TOKEN,
                "edge_instance_id": _SENTINEL_EDGE,
            },
        }
    ).encode("utf-8")

    published = await host._publish_raw_to_dlq(
        original_topic=_ORIGINAL_TOPIC,
        raw_msg=_RawRecord(body),
        error=ValueError("handler rejected the attach"),
        correlation_id=uuid4(),
        failure_type="handler_exception",
        consumer_group="test-group",
    )

    assert published is True
    producer: _CapturingProducer = host._producer  # type: ignore[assignment]
    assert len(producer.sent) == 1

    envelope = json.loads(producer.sent[0]["value"].decode("utf-8"))
    assert _SENTINEL_TOKEN not in producer.sent[0]["value"].decode("utf-8")
    assert envelope["redacted_fields"] == ["payload.access_token"]

    original_body = json.loads(envelope["original_message"]["value"])
    assert original_body["payload"]["access_token"] == DLQ_REDACTION_MARKER
    assert original_body["payload"]["edge_instance_id"] == _SENTINEL_EDGE

    dlq_message = ModelDlqMessage.from_kafka_message(
        envelope,
        dlq_offset=137,
        dlq_partition=0,
    )
    eligible, reason = should_replay(
        dlq_message,
        ModelDlqReplayEngineConfig(
            bootstrap_servers="localhost:9092",
            dlq_topic=_DLQ_TOPIC,
        ),
    )

    assert eligible is False
    assert "payload.access_token" in reason
    assert "OMN-18385" in reason
