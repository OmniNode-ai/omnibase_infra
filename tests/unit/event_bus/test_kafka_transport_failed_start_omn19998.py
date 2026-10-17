# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: a consumer whose start() did not finish is never polled.

Measured on the lab gateway forwarder on 2026-10-16. At 14:45:57Z the
refused-topic retry stopped the inbound consumer and built a new one. From
14:46:03Z every inbound poll failed with ``AttributeError: 'NoneType' object
has no attribute 'check_errors'``, 114 times in a row, and the inbound leg
delivered nothing. aiokafka 0.13.0 ``getmany`` calls
``self._coordinator.check_errors()``, and ``_coordinator`` is only set inside
``start()``. ``_start_consumer`` published the new consumer on the transport
before ``await consumer.start()`` and only cleared it for the two broker
refusals, so a start that raised anything else, or was cancelled, left a
never-started consumer that every later poll called.

The contract pinned here: a start that does not finish leaves no consumer
behind, and the next poll on a started transport builds a fresh one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from aiokafka.errors import KafkaConnectionError, TopicAuthorizationFailedError
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = [pytest.mark.unit]

HEARTBEAT = "tenant-t.onex.evt.omnibase-infra.gateway-heartbeat.v1"
DELEGATION = "tenant-t.onex.cmd.omnibase-infra.delegation-request.v1"
WEBHOOK = "tenant-t.onex.cmd.github.webhook-delivery.v1"
RETRY_SECONDS = 300.0


@dataclass
class _Broker:
    unauthorized: set[str] = field(default_factory=set)
    # Raised by the next consumer start() that passes the refusal checks.
    fail_next_start: BaseException | None = None
    consumers: list[_FakeConsumer] = field(default_factory=list)


class _FakeCluster:
    def __init__(self, broker: _Broker, topics: tuple[str, ...]) -> None:
        self.unauthorized_topics = {t for t in topics if t in broker.unauthorized}

    def partitions_for_topic(self, topic: str) -> set[int] | None:
        return None if topic in self.unauthorized_topics else {0}


class _FakeClient:
    def __init__(self, broker: _Broker, topics: tuple[str, ...]) -> None:
        self.cluster = _FakeCluster(broker, topics)


class _FakeConsumer:
    """Mimics aiokafka: ``getmany`` before ``start()`` finished has no coordinator."""

    broker: _Broker

    def __init__(self, *topics: str, **_kwargs: object) -> None:
        self.topics = tuple(topics)
        self.started = False
        self.stopped = False
        self._client = _FakeClient(self.broker, self.topics)
        self.broker.consumers.append(self)

    async def start(self) -> None:
        for topic in self.topics:
            if topic in self.broker.unauthorized:
                raise TopicAuthorizationFailedError(topic)
        failure = self.broker.fail_next_start
        if failure is not None:
            self.broker.fail_next_start = None
            raise failure
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getmany(
        self, timeout_ms: int = 0, max_records: int | None = None
    ) -> dict[TopicPartition, list[object]]:
        if not self.started and not self.stopped:
            raise AttributeError("'NoneType' object has no attribute 'check_errors'")
        return {}

    def assignment(self) -> set[TopicPartition]:
        return {TopicPartition(t, 0) for t in self.topics} if self.started else set()


class _FakeProducer:
    def __init__(self, **_kwargs: object) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Broker]:
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    state = _Broker()
    consumer_cls = type("_BoundFakeConsumer", (_FakeConsumer,), {"broker": state})
    monkeypatch.setattr(kafka_transport, "AIOKafkaConsumer", consumer_cls)
    monkeypatch.setattr(kafka_transport, "AIOKafkaProducer", _FakeProducer)
    return state


def _transport(clock: _Clock) -> KafkaTransport:
    return KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092"),
        group="tenant-t-gateway-forwarder-inbound",
        topics=(HEARTBEAT, DELEGATION, WEBHOOK),
        refused_topic_retry_seconds=RETRY_SECONDS,
        clock=clock,
    )


def _live(broker: _Broker) -> _FakeConsumer:
    live = [c for c in broker.consumers if c.started and not c.stopped]
    assert len(live) == 1, f"expected exactly one live consumer, got {len(live)}"
    return live[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [KafkaConnectionError("bootstrap timed out"), asyncio.CancelledError()],
    ids=["start-raised", "start-cancelled"],
)
async def test_a_retry_whose_start_did_not_finish_recovers_on_the_next_poll(
    broker: _Broker, failure: BaseException
) -> None:
    """The live sequence: refused-topic retry, then a start that never finishes."""
    broker.unauthorized = {WEBHOOK}
    clock = _Clock()
    transport = _transport(clock)
    await transport.start()
    first = _live(broker)

    # The retry is due; its rebuilt consumer is refused the webhook topic again,
    # peels it, and the start of the two-topic consumer does not finish.
    clock.now += RETRY_SECONDS + 1
    broker.fail_next_start = failure
    with pytest.raises(type(failure)):
        await transport.poll(max_messages=1, timeout_ms=1)
    assert first.stopped is True

    # No never-started consumer is left behind for the watchdog to trust...
    assert transport.has_group_membership() is False
    # ...and the next poll builds a fresh one instead of calling the dead one.
    assert await transport.poll(max_messages=1, timeout_ms=1) == []
    assert set(_live(broker).topics) == {HEARTBEAT, DELEGATION}
    assert transport.refused_topics == frozenset({WEBHOOK})
    assert transport.has_group_membership() is True
    await transport.close()


@pytest.mark.asyncio
async def test_a_healthy_retry_is_unchanged(broker: _Broker) -> None:
    """Positive control: the same retry with a start that finishes."""
    broker.unauthorized = {WEBHOOK}
    clock = _Clock()
    transport = _transport(clock)
    await transport.start()

    clock.now += RETRY_SECONDS + 1
    assert await transport.poll(max_messages=1, timeout_ms=1) == []
    assert set(_live(broker).topics) == {HEARTBEAT, DELEGATION}
    assert transport.has_group_membership() is True
    await transport.close()
