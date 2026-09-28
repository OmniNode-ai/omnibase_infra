# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15629: the shipped inbound leg survives a refused webhook topic.

The unit tests pin KafkaTransport's refused-topic peel with a hand-built
config. These tests resolve the shipped deployment that the .201 gateway
runs, ``docker/gateway/beta-gateway-canary.yaml``, through the real runtime
config loader, and construct its cloud-inbound transport like the forwarder.
On 2026-09-28, the .201 forwarder restarted 207 times when the webhook topic
was refused, taking the admitted inbound topics and heartbeat down with it.
Only aiokafka is faked: the shipped topics and retry policy must keep that
leg alive and restore the refused topic once the broker admits it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiokafka.errors import TopicAuthorizationFailedError, UnknownTopicOrPartitionError
from aiokafka.structs import TopicPartition

from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
)
from omnibase_infra.runtime import gateway_forwarder
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
WEBHOOK = "onex.cmd.github.webhook-delivery.v1"
HEARTBEAT = "onex.evt.omnibase-infra.gateway-heartbeat.v1"
MARKER = "kafka_transport_topic_refused"


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
def broker(monkeypatch: pytest.MonkeyPatch) -> _Broker:
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    state = _Broker()
    consumer_cls = type("_BoundFakeConsumer", (_FakeConsumer,), {"broker": state})
    monkeypatch.setattr(kafka_transport, "AIOKafkaConsumer", consumer_cls)
    monkeypatch.setattr(kafka_transport, "AIOKafkaProducer", _FakeProducer)
    return state


def _shipped_config(
    tmp_path: Path,
) -> ModelGatewayForwarderRuntimeConfig:
    # The shipped config's dev-lane legs authenticate and the loader fails
    # closed without a credential map. Obviously-fake values: this test is
    # about topic and retry policy resolution, not the credential.
    credential_map = tmp_path / "lane-credentials.yaml"
    credential_map.write_text(
        "lane.dev.kafka.scram:\n"
        "  username: fixture-principal\n"
        "  password: fixture-not-a-credential\n",
        encoding="utf-8",
    )
    return gateway_forwarder.load_gateway_forwarder_runtime_config(
        _REPO_ROOT / "docker/gateway/beta-gateway-canary.yaml",
        broker_ref_map_path=_REPO_ROOT
        / "tests/fixtures/gateway/beta-gateway-canary-broker-ref-map.yaml",
        lane_credential_map_path=credential_map,
    )


def _live(broker: _Broker) -> _FakeConsumer:
    live = [c for c in broker.consumers if c.started and not c.stopped]
    assert len(live) == 1, f"expected exactly one live consumer, got {len(live)}"
    return live[0]


@pytest.mark.asyncio
async def test_shipped_inbound_set_carries_the_webhook_topic_and_the_heartbeat(
    tmp_path: Path,
) -> None:
    config = _shipped_config(tmp_path)
    assert config.forwarder.mirror_topics is not None
    slug = config.forwarder.tenant_identity.tenant_slug
    inbound = {prefix_topic(slug, t) for t in config.forwarder.mirror_topics.inbound}

    assert prefix_topic(slug, WEBHOOK) in inbound
    assert prefix_topic(slug, HEARTBEAT) in inbound
    assert config.forwarder.inbound_topic_retry_seconds == 300


@pytest.mark.asyncio
async def test_shipped_inbound_leg_survives_a_refused_webhook_topic(
    tmp_path: Path, broker: _Broker, caplog: pytest.LogCaptureFixture
) -> None:
    config = _shipped_config(tmp_path)
    assert config.cloud_bus is not None
    assert config.forwarder.mirror_topics is not None
    slug = config.forwarder.tenant_identity.tenant_slug
    inbound = {prefix_topic(slug, t) for t in config.forwarder.mirror_topics.inbound}
    webhook = prefix_topic(slug, WEBHOOK)
    heartbeat = prefix_topic(slug, HEARTBEAT)
    broker.unauthorized = {webhook}
    clock = _Clock()
    transport = KafkaTransport(
        config=config.cloud_bus,
        group=f"tenant-{slug}-gateway-forwarder-inbound",
        topics=tuple(
            prefix_topic(slug, t) for t in config.forwarder.mirror_topics.inbound
        ),
        auto_offset_reset=config.cloud_bus.auto_offset_reset,
        refused_topic_retry_seconds=float(config.forwarder.inbound_topic_retry_seconds),
        clock=clock,
    )
    caplog.set_level(logging.WARNING)

    try:
        await transport.start()

        assert set(_live(broker).topics) == inbound - {webhook}
        assert heartbeat in _live(broker).topics
        assert transport.refused_topics == frozenset({webhook})
        assert any(
            record.levelno == logging.WARNING
            and MARKER in record.getMessage()
            and webhook in record.getMessage()
            for record in caplog.records
        )
        assert await transport.poll(max_messages=1, timeout_ms=1) == []

        broker.unauthorized = set()
        clock.now += config.forwarder.inbound_topic_retry_seconds + 1
        assert await transport.poll(max_messages=1, timeout_ms=1) == []

        assert set(_live(broker).topics) == inbound
        assert transport.refused_topics == frozenset()
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_same_leg_without_the_interval_dies_on_the_refusal(
    tmp_path: Path, broker: _Broker
) -> None:
    """The same shipped leg without the opt-in reproduces the pre-fix crash."""
    config = _shipped_config(tmp_path)
    assert config.cloud_bus is not None
    assert config.forwarder.mirror_topics is not None
    slug = config.forwarder.tenant_identity.tenant_slug
    broker.unauthorized = {prefix_topic(slug, WEBHOOK)}
    transport = KafkaTransport(
        config=config.cloud_bus,
        group=f"tenant-{slug}-gateway-forwarder-inbound",
        topics=tuple(
            prefix_topic(slug, t) for t in config.forwarder.mirror_topics.inbound
        ),
        auto_offset_reset=config.cloud_bus.auto_offset_reset,
        clock=_Clock(),
    )

    try:
        with pytest.raises(TopicAuthorizationFailedError):
            await transport.start()
    finally:
        await transport.close()
