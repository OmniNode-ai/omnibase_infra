# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17896 -- the raw DLQ publisher must not fabricate an empty body.

FIRST OF THE TWO SILENT EMPTY DEFAULTS IN SERIES. When a record's value cannot
be read, ``MixinKafkaDlq._publish_raw_to_dlq`` read it as
``getattr(raw_msg, "value", b"")`` and wrote ``"value": ""`` into a DURABLE DLQ
record -- asserting the original message had an empty body when what actually
happened is that the value could not be established. The replay engine then
read that record back, encoded the empty string, and published a ZERO-BYTE
record onto the original topic: 200 of 200 sampled live records on
``onex.evt.omniclaude.tool-executed.v1`` were both zero-byte and
``x-replayed-by=node_dlq_replay_effect`` (dev lane, 2026-09-07).

The precedent for an explicit marker rather than an empty string was already in
this same function: the unreadable-decode case two lines below writes
``"<decode_failed>"``, and the non-serializable fallback writes
``"<non-serializable>"``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.mixin_kafka_dlq import (
    DLQ_UNREADABLE_VALUE_MARKER,
    MixinKafkaDlq,
)
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)

pytestmark = pytest.mark.unit

_DLQ_TOPIC = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: the live DLQ this record lands on
_ORIGINAL_TOPIC = "onex.evt.omniclaude.tool-executed.v1"  # onex-topic-allow: quotes the measured live record


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
        self.sent.append({"topic": topic, "value": value, "key": key})
        return object()


class _DlqHost(MixinKafkaDlq):
    """Minimal host exposing exactly what the mixin declares it needs."""

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


class _ValuelessRecord:
    """A Kafka record object whose value cannot be read -- the live shape the
    silent default was covering for."""

    def __init__(self) -> None:
        self.key = b"k"
        self.offset = 9588531
        self.partition = 0
        self.headers = ()


class _ReadableRecord(_ValuelessRecord):
    def __init__(self) -> None:
        super().__init__()
        self.value = b'{"hello": "world"}'


async def _publish(raw_msg: object) -> dict[str, Any]:
    host = _DlqHost()
    published = await host._publish_raw_to_dlq(
        original_topic=_ORIGINAL_TOPIC,
        raw_msg=raw_msg,
        error=ValueError("deserialization failed"),
        correlation_id=uuid4(),
        failure_type="deserialization_error",
        consumer_group="test-group",
    )
    assert published is True
    producer: _CapturingProducer = host._producer  # type: ignore[assignment]
    assert len(producer.sent) == 1, producer.sent
    payload = json.loads(producer.sent[0]["value"].decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


async def test_3_an_unreadable_value_is_marked_not_emptied() -> None:
    """RED on origin/dev: ``original_message.value`` is written as ``""``."""
    payload = await _publish(_ValuelessRecord())
    written = payload["original_message"]["value"]
    assert written != "", (
        "the DLQ record asserted the original message had an EMPTY body; what "
        "actually happened is that the value could not be read, and the replay "
        "engine then published that fabricated empty body as a zero-byte record"
    )
    assert written == DLQ_UNREADABLE_VALUE_MARKER, written


async def test_3b_a_readable_value_is_still_written_verbatim() -> None:
    """Positive control for (3): the marker is not written unconditionally."""
    payload = await _publish(_ReadableRecord())
    assert payload["original_message"]["value"] == '{"hello": "world"}'
