# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural conformance of ``KafkaTransport`` to the core transport protocols.

Broker-free (no network): proves the Kafka face satisfies both S1 protocols
(``ProtocolTransportConsumer`` / ``ProtocolTransportProducer``, epic OMN-14717) at
the type/attribute level, and that the two S3 requirements are honored — the
constructor forces neither auto-commit nor a subscribe surface, and the consumer is
built with ``enable_auto_commit=False`` while the legacy ``EventBusKafka`` push
consumers keep their per-consumer config setting. The observable Kafka semantics
(commit-all-<=-k, nack-replay, restart-redeliver) are proven against a live broker by
the parametrized conformance suite in ``tests/integration/transport``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import omnibase_infra.event_bus.kafka_transport as kafka_transport_module
from omnibase_core.protocols.runtime.protocol_transport_consumer import (
    ProtocolTransportConsumer,
)
from omnibase_core.protocols.runtime.protocol_transport_producer import (
    ProtocolTransportProducer,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.topic_constants import build_dlq_topic


@pytest.fixture
def unit_bootstrap() -> str:
    return ModelKafkaEventBusConfig.default().bootstrap_servers


@pytest.fixture
def unit_topic() -> str:
    return build_dlq_topic("events")


def _make_transport(unit_bootstrap: str, unit_topic: str) -> KafkaTransport:
    # Construction only — never started, so no broker connection is attempted.
    return KafkaTransport.from_bootstrap(
        unit_bootstrap, group="unit-conformance", topics=[unit_topic]
    )


def test_kafka_transport_is_a_transport_consumer(
    unit_bootstrap: str, unit_topic: str
) -> None:
    transport = _make_transport(unit_bootstrap, unit_topic)
    assert isinstance(transport, ProtocolTransportConsumer)
    # Assignable to the protocol type (mypy contravariance check lives in CI).
    consumer: ProtocolTransportConsumer = transport
    assert consumer is transport


def test_kafka_transport_is_a_transport_producer(
    unit_bootstrap: str, unit_topic: str
) -> None:
    transport = _make_transport(unit_bootstrap, unit_topic)
    assert isinstance(transport, ProtocolTransportProducer)
    producer: ProtocolTransportProducer = transport
    assert producer is transport


def test_poll_commit_nack_send_signatures_are_async(
    unit_bootstrap: str, unit_topic: str
) -> None:
    transport = _make_transport(unit_bootstrap, unit_topic)
    for name in ("start", "close", "poll", "commit", "nack", "send"):
        method = getattr(transport, name)
        assert inspect.iscoroutinefunction(method), f"{name} must be async"


def test_from_bootstrap_overrides_bootstrap_servers(unit_bootstrap: str) -> None:
    transport = KafkaTransport.from_bootstrap(unit_bootstrap)
    assert transport._config.bootstrap_servers == unit_bootstrap
    # Pull-based at-least-once consumer defaults to earliest so a fresh group sees
    # the existing backlog on first boot (required by the conformance suite).
    assert transport._auto_offset_reset == "earliest"


def test_producer_only_instance_has_no_topics(unit_bootstrap: str) -> None:
    transport = KafkaTransport.from_bootstrap(unit_bootstrap)
    assert transport._topics == ()


def test_legacy_eventbuskafka_consumer_setting_untouched() -> None:
    # S3 scope guard: forcing enable_auto_commit=False applies to the NEW transport
    # consumers only. EventBusKafka still reads its per-consumer config default
    # (True) for the legacy push path — this face changes nothing about it.
    bus = EventBusKafka.default()
    assert bus._config.enable_auto_commit is True


def test_to_model_rejects_nullable_kafka_headers(unit_topic: str) -> None:
    record = SimpleNamespace(
        topic=unit_topic,
        partition=0,
        offset=7,
        key=None,
        value=b"payload",
        headers=[("nullable", None)],
    )

    with pytest.raises(ProtocolConfigurationError, match="nullable Kafka header"):
        KafkaTransport._to_model(SimpleNamespace(topic=unit_topic, partition=0), record)


@pytest.mark.asyncio
async def test_start_rolls_back_producer_when_consumer_start_fails(
    monkeypatch: pytest.MonkeyPatch, unit_bootstrap: str, unit_topic: str
) -> None:
    stopped: list[str] = []

    class FakeProducer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            stopped.append("producer")

    class FakeConsumer:
        def __init__(self, *_topics: str, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            raise RuntimeError("consumer boom")

        async def stop(self) -> None:
            stopped.append("consumer")

    monkeypatch.setattr(kafka_transport_module, "AIOKafkaProducer", FakeProducer)
    monkeypatch.setattr(kafka_transport_module, "AIOKafkaConsumer", FakeConsumer)

    transport = KafkaTransport.from_bootstrap(
        unit_bootstrap, group="unit-conformance", topics=[unit_topic]
    )

    with pytest.raises(RuntimeError, match="consumer boom"):
        await transport.start()

    assert stopped == ["consumer", "producer"]
    assert transport._consumer is None
    assert transport._producer is None
    assert transport._started is False


@pytest.mark.asyncio
async def test_close_attempts_producer_stop_after_consumer_stop_fails(
    unit_bootstrap: str,
) -> None:
    stopped: list[str] = []

    class FailingConsumer:
        async def stop(self) -> None:
            stopped.append("consumer")
            raise RuntimeError("consumer stop boom")

    class StoppingProducer:
        async def stop(self) -> None:
            stopped.append("producer")

    transport = KafkaTransport.from_bootstrap(unit_bootstrap)
    transport._consumer = FailingConsumer()  # type: ignore[assignment]
    transport._producer = StoppingProducer()  # type: ignore[assignment]
    transport._started = True

    with pytest.raises(RuntimeError, match="consumer stop boom"):
        await transport.close()

    assert stopped == ["consumer", "producer"]
    assert transport._consumer is None
    assert transport._producer is None
    assert transport._started is False
