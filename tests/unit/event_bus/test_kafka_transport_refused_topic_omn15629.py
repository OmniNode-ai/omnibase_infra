# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One refused subscribe topic must not kill a whole consumer (OMN-15629).

Measured on the .201 dev lane gateway forwarder, 2026-09-28 10:31:59Z onward.
omnibase_infra#4227 added the tenant GitHub webhook-delivery command topic to
the forwarder's inbound set. That topic cannot exist on the cloud broker until
the cloud owner places the webhook secret (OMN-19592), so aiokafka's
``AIOKafkaConsumer.start`` -> ``_wait_topics`` raised
``TopicAuthorizationFailedError`` for it. ``KafkaTransport.start`` let that
propagate, ``run_gateway_forwarder`` died before the delivery loop or the
heartbeat task existed, and the container restarted 114 times. Every other
inbound topic and the local heartbeat mirror died with it, which is what
turned the M4 C28 consumer-flow canary red: the heartbeat projection counted
``messages_in`` 0.

The contract these tests pin, for a transport constructed with
``refused_topic_retry_seconds``:

* a subscribe topic the broker refuses (not authorized, or missing) is peeled
  off the subscription and the consumer starts on the rest;
* the refusal is visible: a WARNING carrying the stable marker
  ``kafka_transport_topic_refused`` and the canonical topic name, and the
  topic is readable from ``refused_topics``;
* the refused topic is retried on the interval, and rejoins the subscription
  the first retry after the broker admits it;
* a transport constructed WITHOUT the interval keeps today's fail-fast start,
  so no other caller silently loses a topic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from aiokafka.errors import TopicAuthorizationFailedError, UnknownTopicOrPartitionError
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = [pytest.mark.unit]

HEARTBEAT = "tenant-t.onex.evt.omnibase-infra.gateway-heartbeat.v1"
DELEGATION = "tenant-t.onex.cmd.omnibase-infra.delegation-request.v1"
WEBHOOK = "tenant-t.onex.cmd.github.webhook-delivery.v1"
MARKER = "kafka_transport_topic_refused"
RETRY_SECONDS = 300.0


@dataclass
class _Broker:
    """What the fake broker refuses, and every consumer it was asked for."""

    unauthorized: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    consumers: list[_FakeConsumer] = field(default_factory=list)


class _FakeCluster:
    def __init__(self, broker: _Broker, topics: tuple[str, ...]) -> None:
        self.unauthorized_topics = {t for t in topics if t in broker.unauthorized}
        self._known = {
            t
            for t in topics
            if t not in broker.unauthorized and t not in broker.missing
        }

    def partitions_for_topic(self, topic: str) -> set[int] | None:
        return {0} if topic in self._known else None


class _FakeClient:
    def __init__(self, broker: _Broker, topics: tuple[str, ...]) -> None:
        self.cluster = _FakeCluster(broker, topics)


class _FakeConsumer:
    """Mimics ``AIOKafkaConsumer.start`` -> ``_wait_topics`` refusal semantics."""

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
            if topic in self.broker.missing:
                # aiokafka raises this one with no topic argument.
                raise UnknownTopicOrPartitionError
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getmany(
        self, timeout_ms: int = 0, max_records: int | None = None
    ) -> dict[TopicPartition, list[object]]:
        return {}

    def assignment(self) -> set[TopicPartition]:
        return {TopicPartition(t, 0) for t in self.topics} if self.started else set()


class _FakeProducer:
    def __init__(self, **_kwargs: object) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


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


def _transport(
    *,
    retry_seconds: float | None = RETRY_SECONDS,
    clock: _Clock | None = None,
    topics: tuple[str, ...] = (HEARTBEAT, DELEGATION, WEBHOOK),
) -> KafkaTransport:
    kwargs: dict[str, object] = {}
    if retry_seconds is not None:
        kwargs["refused_topic_retry_seconds"] = retry_seconds
    if clock is not None:
        kwargs["clock"] = clock
    return KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092"),
        group="tenant-t-gateway-forwarder-inbound",
        topics=topics,
        **kwargs,  # type: ignore[arg-type]
    )


def _live(broker: _Broker) -> _FakeConsumer:
    live = [c for c in broker.consumers if c.started and not c.stopped]
    assert len(live) == 1, f"expected exactly one live consumer, got {len(live)}"
    return live[0]


def _refusal_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and MARKER in r.getMessage()
    ]


@pytest.mark.asyncio
async def test_one_unauthorized_topic_does_not_kill_the_consumer(
    broker: _Broker, caplog: pytest.LogCaptureFixture
) -> None:
    """The RED reproduction: this is the 114-restart crash on the dev lane."""
    broker.unauthorized = {WEBHOOK}
    transport = _transport(clock=_Clock())
    caplog.set_level(logging.WARNING)

    await transport.start()

    assert set(_live(broker).topics) == {HEARTBEAT, DELEGATION}
    assert transport.refused_topics == frozenset({WEBHOOK})
    warnings = _refusal_warnings(caplog)
    assert len(warnings) == 1
    assert WEBHOOK in warnings[0]
    assert "TopicAuthorizationFailedError" in warnings[0]
    # The loop keeps polling the topics that were admitted.
    assert await transport.poll(max_messages=1, timeout_ms=1) == []
    assert transport.has_group_membership() is True
    await transport.close()


@pytest.mark.asyncio
async def test_a_missing_topic_is_peeled_the_same_way(
    broker: _Broker, caplog: pytest.LogCaptureFixture
) -> None:
    broker.missing = {WEBHOOK}
    transport = _transport(clock=_Clock())
    caplog.set_level(logging.WARNING)

    await transport.start()

    assert set(_live(broker).topics) == {HEARTBEAT, DELEGATION}
    assert transport.refused_topics == frozenset({WEBHOOK})
    warnings = _refusal_warnings(caplog)
    assert len(warnings) == 1
    assert "UnknownTopicOrPartitionError" in warnings[0]
    await transport.close()


@pytest.mark.asyncio
async def test_refused_topic_rejoins_on_the_first_retry_after_it_is_admitted(
    broker: _Broker, caplog: pytest.LogCaptureFixture
) -> None:
    broker.unauthorized = {WEBHOOK}
    clock = _Clock()
    transport = _transport(clock=clock)
    await transport.start()
    first = _live(broker)

    # Before the interval: no retry, no churn, even though the ACL landed.
    broker.unauthorized = set()
    clock.now += RETRY_SECONDS - 1
    await transport.poll(max_messages=1, timeout_ms=1)
    assert _live(broker) is first

    clock.now += 2
    caplog.set_level(logging.INFO)
    await transport.poll(max_messages=1, timeout_ms=1)

    assert set(_live(broker).topics) == {HEARTBEAT, DELEGATION, WEBHOOK}
    assert first.stopped is True
    assert transport.refused_topics == frozenset()
    await transport.close()


@pytest.mark.asyncio
async def test_a_still_refused_topic_warns_again_and_the_rest_keep_flowing(
    broker: _Broker, caplog: pytest.LogCaptureFixture
) -> None:
    broker.unauthorized = {WEBHOOK}
    clock = _Clock()
    transport = _transport(clock=clock)
    caplog.set_level(logging.WARNING)
    await transport.start()

    clock.now += RETRY_SECONDS + 1
    await transport.poll(max_messages=1, timeout_ms=1)

    assert set(_live(broker).topics) == {HEARTBEAT, DELEGATION}
    assert transport.refused_topics == frozenset({WEBHOOK})
    assert len(_refusal_warnings(caplog)) == 2
    await transport.close()


@pytest.mark.asyncio
async def test_every_topic_refused_leaves_the_transport_alive_and_idle(
    broker: _Broker,
) -> None:
    broker.unauthorized = {WEBHOOK}
    transport = _transport(clock=_Clock(), topics=(WEBHOOK,))

    await transport.start()

    assert [c for c in broker.consumers if c.started and not c.stopped] == []
    assert transport.refused_topics == frozenset({WEBHOOK})
    assert await transport.poll(max_messages=1, timeout_ms=1) == []
    # Nothing to be a member FOR: the membership watchdog must not thrash.
    assert transport.has_group_membership() is True
    await transport.close()


@pytest.mark.asyncio
async def test_restart_consumer_keeps_peeling(broker: _Broker) -> None:
    """The delivery watchdog's recreate path goes through the same peel."""
    broker.unauthorized = {WEBHOOK}
    transport = _transport(clock=_Clock())
    await transport.start()

    await transport.restart_consumer()

    assert set(_live(broker).topics) == {HEARTBEAT, DELEGATION}
    assert transport.refused_topics == frozenset({WEBHOOK})
    await transport.close()


@pytest.mark.asyncio
async def test_without_the_interval_start_still_fails_fast(broker: _Broker) -> None:
    """Opt-in only: every other KafkaTransport caller keeps today's semantics."""
    broker.unauthorized = {WEBHOOK}
    transport = _transport(retry_seconds=None)

    with pytest.raises(TopicAuthorizationFailedError):
        await transport.start()
