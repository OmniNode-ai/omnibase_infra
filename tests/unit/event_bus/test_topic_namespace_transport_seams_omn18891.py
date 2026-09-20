# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""The deployment topic namespace is applied at the transport seams (OMN-18891).

A namespace that is configured but not applied at the wire is worse than no
namespace at all: the runtime reports isolation it does not have. These tests
read the seam itself — the arguments handed to the broker client — rather than
a log line or a return value the code under test chose to produce.

The two directions are asserted separately because they fail differently. A
missing PUBLISH prefix puts this runtime's events onto the shared topic the
dev lane consumes. A missing SUBSCRIBE prefix hands this runtime a full copy
of every event the dev lane produces. Both defeat the isolation; only the
first is visible from outside.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = [pytest.mark.unit]

BOOTSTRAP = "localhost:9092"
CANONICAL = "onex.evt.platform.node-registration.v1"
SECOND_CANONICAL = "onex.cmd.platform.request-introspection.v1"
SLOT = "prepr1"


def _bus() -> EventBusKafka:
    return EventBusKafka(config=ModelKafkaEventBusConfig(bootstrap_servers=BOOTSTRAP))


def _transport(topics: tuple[str, ...] = (CANONICAL,)) -> KafkaTransport:
    return KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers=BOOTSTRAP),
        group="onex.transport.kafka",
        topics=topics,
    )


# ---------------------------------------------------------------------------
# EventBusKafka: the subscribe seam
# ---------------------------------------------------------------------------


def test_bus_consumer_subscribes_to_the_canonical_topic_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer"
    ) as consumer_cls:
        _bus()._build_consumer(CANONICAL, "g", "gi", "earliest")
    assert consumer_cls.call_args.args[0] == CANONICAL


def test_bus_consumer_subscribes_to_the_physical_topic_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer"
    ) as consumer_cls:
        _bus()._build_consumer(CANONICAL, "g", "gi", "earliest")
    assert consumer_cls.call_args.args[0] == f"{SLOT}.{CANONICAL}"


# ---------------------------------------------------------------------------
# KafkaTransport: subscribe, publish, and the canonical round trip
# ---------------------------------------------------------------------------


def test_transport_keeps_canonical_topics_and_derives_physical_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime's view stays canonical; only the wire view moves."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    transport = _transport((CANONICAL, SECOND_CANONICAL))
    assert transport._topics == (CANONICAL, SECOND_CANONICAL)
    assert transport._physical_topics == (
        f"{SLOT}.{CANONICAL}",
        f"{SLOT}.{SECOND_CANONICAL}",
    )


def test_transport_physical_equals_canonical_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    transport = _transport((CANONICAL, SECOND_CANONICAL))
    assert transport._physical_topics == transport._topics


@pytest.mark.asyncio
async def test_transport_send_publishes_physical_and_answers_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    transport = _transport()

    sent: dict[str, Any] = {}

    class _Metadata:
        topic = f"{SLOT}.{CANONICAL}"
        partition = 3
        offset = 17

    async def _send_and_wait(topic: str, **kwargs: Any) -> _Metadata:
        sent["topic"] = topic
        return _Metadata()

    producer = MagicMock()
    producer.send_and_wait = _send_and_wait
    transport._producer = producer

    coordinate = await transport.send_with_coordinate(CANONICAL, None, b"payload", {})

    assert sent["topic"] == f"{SLOT}.{CANONICAL}", "published onto the shared topic"
    assert coordinate == (CANONICAL, 3, 17), "coordinate must answer in canonical terms"


@pytest.mark.asyncio
async def test_transport_send_is_unchanged_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    transport = _transport()

    sent: dict[str, Any] = {}

    class _Metadata:
        topic = CANONICAL
        partition = 0
        offset = 1

    async def _send_and_wait(topic: str, **kwargs: Any) -> _Metadata:
        sent["topic"] = topic
        return _Metadata()

    producer = MagicMock()
    producer.send_and_wait = _send_and_wait
    transport._producer = producer

    assert await transport.send_with_coordinate(CANONICAL, None, b"p", {}) == (
        CANONICAL,
        0,
        1,
    )
    assert sent["topic"] == CANONICAL


# ---------------------------------------------------------------------------
# The mapping control: a physical name must not reach a comparison or a lookup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caught_up_probe_answers_in_canonical_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The assignment is physical and the caller asks in canonical names.

    Without the mapping this probe reports every topic as not-caught-up, which
    reads as a phantom subscription rather than as a namespace bug.
    """
    from aiokafka import TopicPartition

    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    transport = _transport()

    physical_tp = TopicPartition(f"{SLOT}.{CANONICAL}", 0)
    consumer = MagicMock()
    consumer.assignment.return_value = {physical_tp}
    consumer.highwater.return_value = 5

    async def _position(_tp: object) -> int:
        return 5

    consumer.position = _position
    transport._consumer = consumer

    assert await transport.caught_up_topics(frozenset({CANONICAL})) == frozenset(
        {CANONICAL}
    )
