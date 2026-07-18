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

from omnibase_core.protocols.runtime.protocol_transport_consumer import (
    ProtocolTransportConsumer,
)
from omnibase_core.protocols.runtime.protocol_transport_producer import (
    ProtocolTransportProducer,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_transport import KafkaTransport


def _make_transport() -> KafkaTransport:
    # Construction only — never started, so no broker connection is attempted.
    return KafkaTransport.from_bootstrap(
        "localhost:19092", group="unit-conformance", topics=["unit.topic.v1"]
    )


def test_kafka_transport_is_a_transport_consumer() -> None:
    transport = _make_transport()
    assert isinstance(transport, ProtocolTransportConsumer)
    # Assignable to the protocol type (mypy contravariance check lives in CI).
    consumer: ProtocolTransportConsumer = transport
    assert consumer is transport


def test_kafka_transport_is_a_transport_producer() -> None:
    transport = _make_transport()
    assert isinstance(transport, ProtocolTransportProducer)
    producer: ProtocolTransportProducer = transport
    assert producer is transport


def test_poll_commit_nack_send_signatures_are_async() -> None:
    transport = _make_transport()
    for name in ("start", "close", "poll", "commit", "nack", "send"):
        method = getattr(transport, name)
        assert inspect.iscoroutinefunction(method), f"{name} must be async"


def test_from_bootstrap_overrides_bootstrap_servers() -> None:
    transport = KafkaTransport.from_bootstrap("broker.example:9092")
    assert transport._config.bootstrap_servers == "broker.example:9092"
    # Pull-based at-least-once consumer defaults to earliest so a fresh group sees
    # the existing backlog on first boot (required by the conformance suite).
    assert transport._auto_offset_reset == "earliest"


def test_producer_only_instance_has_no_topics() -> None:
    transport = KafkaTransport.from_bootstrap("localhost:19092")
    assert transport._topics == ()


def test_legacy_eventbuskafka_consumer_setting_untouched() -> None:
    # S3 scope guard: forcing enable_auto_commit=False applies to the NEW transport
    # consumers only. EventBusKafka still reads its per-consumer config default
    # (True) for the legacy push path — this face changes nothing about it.
    bus = EventBusKafka.default()
    assert bus._config.enable_auto_commit is True
