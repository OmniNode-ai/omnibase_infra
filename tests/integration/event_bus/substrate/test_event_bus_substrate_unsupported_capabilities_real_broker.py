# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""AC4 (real_broker half): unsupported-list capabilities against a live broker.

``EventBusSemanticFake`` (omnibase_core) explicitly RAISES for six broker
behaviors it does not model (see that module's docstring). AC4 requires
"the identical test passes on real_broker" for each. This file is the
honest, capability-by-capability accounting of that half:

- **Provable through ``EventBusKafka``'s current surface** (implemented
  below): multi-partition, key-based ordering. A real broker genuinely
  routes same-key messages to the same partition and preserves per-key
  order; ``EventBusSemanticFake`` refuses this outright
  (``partition_count != 1`` raises at construction,
  ``publish_to_partition()`` always raises).

- **NOT provable through ``EventBusKafka``'s current surface** (explicitly
  skipped below, each with its own reason -- not silently omitted):
  exactly-once/transactional producer semantics, consumer lag/backpressure,
  broker failover/leader election, compacted-topic tombstones,
  wire-protocol/codec configuration. ``EventBusKafka`` does not expose
  producer transactions, per-consumer lag querying, failover simulation,
  tombstone publishing, or compression/batching configuration as first-class
  operations today -- proving these needs either extending ``EventBusKafka``
  itself (out of this ticket's "reuse the live EventBusKafka, no new impl"
  scope) or driving the underlying ``aiokafka`` client directly, bypassing
  ``ProtocolEventBus`` entirely (which would test aiokafka, not this
  fixture's substrate contract). Recorded here as a known, named gap rather
  than a fabricated pass.

All tests in this file are ``pytest.mark.integration`` + ``pytest.mark.kafka``,
gated by this directory's ``conftest.py`` on ``KAFKA_INTEGRATION_TESTS=1``.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.event_bus.event_bus_semantic_fake import EventBusSemanticFake
from tests.helpers.util_kafka import KafkaTopicManager

if TYPE_CHECKING:
    from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

pytestmark = [pytest.mark.integration, pytest.mark.kafka]


@dataclass
class _PartitionOrderingIdentity:
    env: str = "test"
    service: str = "event-bus-substrate"
    node_name: str = "partition-ordering-probe"
    version: str = "v1"


def _real_broker_available() -> bool:
    return os.getenv("KAFKA_INTEGRATION_TESTS") == "1"


@pytest.mark.asyncio
async def test_semantic_fake_refuses_multi_partition_construction() -> None:
    """The fake half: constructing with partition_count != 1 always raises."""
    with pytest.raises(ModelOnexError):
        EventBusSemanticFake(partition_count=3)


@pytest.mark.asyncio
async def test_real_broker_preserves_per_key_partition_ordering() -> None:
    """The real_broker half: same-key messages land on the same partition,
    in publish order -- the exact guarantee EventBusSemanticFake refuses to
    model (single partition only).
    """
    if not _real_broker_available():
        pytest.skip(
            "requires KAFKA_INTEGRATION_TESTS=1 and a reachable broker "
            "(see this directory's conftest.py)"
        )

    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]
    topic = f"onex.evt.omnibase-infra.partition-order-probe.{uuid.uuid4().hex[:8]}.v1"

    async with KafkaTopicManager(bootstrap_servers) as manager:
        await manager.create_topic(topic, partitions=3, replication_factor=1)

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=bootstrap_servers,
        environment="test",
        timeout_seconds=30,
    )
    bus: EventBusKafka = EventBusKafka(config=config)
    await bus.start()
    try:
        received: list[ModelEventMessage] = []

        async def handler(msg: ModelEventMessage) -> None:
            received.append(msg)

        await bus.subscribe(topic, _PartitionOrderingIdentity(), handler)

        # Interleave two keys across several publishes. Each key's messages
        # must arrive in publish order relative to EACH OTHER (per-partition
        # ordering); global interleaving across keys is not asserted (Kafka
        # makes no such guarantee, and neither does this test).
        key_a = b"entity-a"
        key_b = b"entity-b"
        for i in range(5):
            await bus.publish(topic, key_a, f"a-{i}".encode())
            await bus.publish(topic, key_b, f"b-{i}".encode())

        deadline = asyncio.get_event_loop().time() + 20.0
        while len(received) < 10 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.1)

        assert len(received) == 10, (
            f"expected all 10 keyed messages, got {len(received)}: "
            f"{[m.value for m in received]}"
        )

        a_values = [m.value.decode() for m in received if m.key == key_a]
        b_values = [m.value.decode() for m in received if m.key == key_b]
        assert a_values == [f"a-{i}" for i in range(5)], (
            "key_a messages must arrive in publish order -- a real broker "
            "routes same-key messages to the same partition and preserves "
            "order within it."
        )
        assert b_values == [f"b-{i}" for i in range(5)], (
            "key_b messages must arrive in publish order, independently of "
            "key_a's interleaved partition."
        )
    finally:
        await bus.close()


class TestUnsupportedCapabilitiesNotProvableThroughEventBusKafka:
    """Named, honest gaps -- not silently omitted from AC4's real_broker half.

    Each of these needs either an EventBusKafka extension (transactions,
    lag query, tombstones, wire-codec config) or live chaos engineering
    (broker failover) to prove for real. Out of this ticket's scope
    ("reuse the live EventBusKafka... rather than a new impl") -- flagged
    here as a followup rather than faked green.
    """

    @pytest.mark.skip(
        reason="EventBusKafka exposes no producer-transaction API "
        "(begin_transaction/commit_transaction) today; proving exactly-once "
        "semantics needs either extending EventBusKafka or driving the "
        "underlying aiokafka producer's transactional API directly, "
        "bypassing ProtocolEventBus. OMN-15789 scope is 'reuse the live "
        "EventBusKafka, no new impl' -- tracked as a known gap, not faked."
    )
    def test_transactional_publish_real_broker(self) -> None:
        raise AssertionError("unreachable -- skipped")

    @pytest.mark.skip(
        reason="EventBusKafka.health_check() does not report per-consumer "
        "lag (no consumer.highwater()/position() plumbing); proving this "
        "needs an EventBusKafka extension, out of this ticket's reuse-only "
        "scope."
    )
    def test_consumer_lag_query_real_broker(self) -> None:
        raise AssertionError("unreachable -- skipped")

    @pytest.mark.skip(
        reason="Broker failover/leader-election/ISR shrink-expand requires "
        "live chaos engineering (killing/restarting a broker process) "
        "against real infrastructure -- explicitly out of scope for a "
        "fixture-level contract test, and risky to automate against any "
        "shared broker."
    )
    def test_broker_failover_real_broker(self) -> None:
        raise AssertionError("unreachable -- skipped")

    @pytest.mark.skip(
        reason="EventBusKafka exposes no tombstone-specific publish path; "
        "compacted-topic tombstone semantics also require a compacted "
        "topic (cleanup.policy=compact), which this fixture's ad-hoc test "
        "topics are not configured as. Needs an EventBusKafka extension, "
        "out of this ticket's reuse-only scope."
    )
    def test_tombstone_publish_real_broker(self) -> None:
        raise AssertionError("unreachable -- skipped")

    @pytest.mark.skip(
        reason="EventBusKafka exposes no wire-protocol/codec configuration "
        "surface (batching, compression) on ProtocolEventBus; these are "
        "AIOKafkaProducer constructor-level settings today, not a runtime "
        "call. Needs an EventBusKafka extension, out of this ticket's "
        "reuse-only scope."
    )
    def test_wire_codec_config_real_broker(self) -> None:
        raise AssertionError("unreachable -- skipped")
