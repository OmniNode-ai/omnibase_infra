# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka side of the transport substitutability oracle (OMN-14756, epic OMN-14717).

Runs the SAME parametrized ``TransportConformanceSuite`` that certifies the
in-memory transport in core (S2, ticket OMN-14720) against the real Kafka face
``KafkaTransport`` (S3). The suite asserting identical observable Kafka semantics
against BOTH impls is what licenses "in-memory golden chain => Kafka golden chain":

* poll returns messages in per-partition offset order;
* commit at offset k commits ALL offsets <= k on that partition (the corrected
  Kafka-HWM assertion, plan HOLE 2);
* nack redelivers from the offset (the message + later same-partition offsets);
* uncommitted offsets redeliver on restart (new consumer, same group);
* send is per-topic ordered and headers/keys/partition/offset round-trip.

The suite pins a single fixed topic (``CONFORMANCE_TOPIC``) and documents that each
test gets a "fresh Kafka topic"; the autouse ``_fresh_topic`` fixture delivers that
by dropping and recreating the topic per test so offsets and committed cursors reset.

Broker-gated: skipped when no broker is reachable (see ``_kafka_env``). Intended to
run locally / on ``.201`` stability-test against the external broker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence

import pytest

from omnibase_core.runtime.transport.runtime_transport_conformance import (
    CONFORMANCE_TOPIC,
    TransportConformanceSuite,
)
from omnibase_infra.event_bus.kafka_transport import KafkaTransport

from ._kafka_env import kafka_available, recreate_topic, transport_bootstrap

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
    pytest.mark.heavy,
    pytest.mark.skipif(
        not kafka_available(),
        reason=(
            "no Kafka broker reachable (set ONEX_TRANSPORT_KAFKA_BOOTSTRAP or "
            "KAFKA_BOOTSTRAP_SERVERS to a live broker)"
        ),
    ),
]


class TestKafkaTransportConformance(TransportConformanceSuite):
    """Parametrize the shared conformance suite over the real Kafka transport.

    Mirrors ``TestInMemoryTransportConformance`` in core: same base suite, same two
    fixtures (``transport_producer`` + ``transport_consumer_factory``), Kafka-backed.
    """

    @pytest.fixture
    def _bootstrap(self) -> str:
        return transport_bootstrap()

    @pytest.fixture(autouse=True)
    async def _fresh_topic(self, _bootstrap: str) -> None:
        # A fresh topic per test resets offsets + committed cursors, matching the
        # suite's documented "fresh Kafka topic" isolation. Provisioned with >=2
        # partitions so the shared suite's multi-partition nack test
        # (test_nack_on_one_partition_does_not_strand_sibling_partitions, OMN-14757)
        # has real sibling partitions to strand on the real broker; the same-key
        # single-partition tests still land on one partition, so their assertions
        # are unaffected.
        await recreate_topic(_bootstrap, CONFORMANCE_TOPIC, partitions=2)

    @pytest.fixture
    async def transport_producer(
        self, _bootstrap: str, _fresh_topic: None
    ) -> AsyncIterator[KafkaTransport]:
        # The suite calls ``producer.send(...)`` without starting it, so the fixture
        # yields an already-started producer (topics=() => producer only).
        producer = KafkaTransport.from_bootstrap(_bootstrap)
        await producer.start()
        try:
            yield producer
        finally:
            await producer.close()

    @pytest.fixture
    async def transport_consumer_factory(
        self, _bootstrap: str, _fresh_topic: None
    ) -> AsyncIterator[Callable[..., KafkaTransport]]:
        created: list[KafkaTransport] = []

        def factory(*, group: str, topics: Sequence[str]) -> KafkaTransport:
            # Not started here: the suite calls ``await consumer.start()`` itself.
            transport = KafkaTransport.from_bootstrap(
                _bootstrap, group=group, topics=list(topics)
            )
            created.append(transport)
            return transport

        try:
            yield factory
        finally:
            # Defensive cleanup: the suite closes consumers itself, but a failing
            # test may leave one open.
            for transport in created:
                await transport.close()
