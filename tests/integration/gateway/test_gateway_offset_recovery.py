# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-broker proof of the OMN-15781 offset-policy fix.

Drives the actual seam the gateway forwarder's legs run over
(``KafkaTransport`` against a live Kafka/Redpanda broker), not a mock: a
message is produced to a topic BEFORE any consumer with the gateway's
consumer-group name has ever joined that group, then a fresh
``KafkaTransport`` is started on that group with ``auto_offset_reset``
matching the deployed gateway policy. With ``"earliest"`` (the OMN-15781
fix, and the only value ``ModelGatewayForwarderRuntimeConfig`` now accepts)
the message MUST be delivered. With ``"latest"`` (the pre-fix
``beta-gateway-canary.yaml`` value) it is silently skipped -- the exact
mechanism that lost the 28 ``delegation-completed.v1`` events during the
2026-08-04 outage investigation (OMN-15742).

Broker-gated: skipped when no broker is reachable (see
``tests.integration.transport._kafka_env``), matching the pattern used
there. Intended to run locally / on ``.201``/``.200`` against a live broker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from tests.integration.transport._kafka_env import kafka_available, transport_bootstrap

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


@pytest.fixture
def bootstrap() -> str:
    return transport_bootstrap()


@pytest.fixture
def topic() -> str:
    # Uniquely named per test run (not delete/recreate) -- this suite runs
    # against a shared LAN broker; auto-created via the producer's first
    # ``send`` rather than deleting/recreating a fixed topic name.
    return f"onex.transport.test.gateway-offset-recovery.{uuid4().hex}.v1"


async def _produce_one(bootstrap: str, *, topic: str, value: bytes) -> None:
    producer = KafkaTransport.from_bootstrap(bootstrap)
    await producer.start()
    try:
        await producer.send(topic, None, value, {})
    finally:
        await producer.close()


@pytest.fixture
async def consumer_factory() -> AsyncIterator[list[KafkaTransport]]:
    created: list[KafkaTransport] = []
    yield created
    for transport in created:
        await transport.close()


class TestGatewayOffsetRecoveryRealBroker:
    """OMN-15781: the exact seam the gateway forwarder's legs run over."""

    async def test_earliest_delivers_message_produced_before_group_joined(
        self,
        bootstrap: str,
        topic: str,
        consumer_factory: list[KafkaTransport],
    ) -> None:
        """The OMN-15781 fix: a fresh group on ``earliest`` sees the backlog.

        Mirrors a gateway leg restarting after an outage window (crash,
        LeaveGroup, cold restart) during which the local publisher kept
        producing: the consumer group is brand new (has never committed an
        offset on this topic), and a message already sits at offset 0
        before the consumer ever calls ``start()``.
        """
        group = f"gw-offset-recovery-earliest-{uuid4().hex}"
        await _produce_one(
            bootstrap, topic=topic, value=b"produced-before-group-joined"
        )

        consumer = KafkaTransport.from_bootstrap(
            bootstrap, group=group, topics=[topic], auto_offset_reset="earliest"
        )
        consumer_factory.append(consumer)
        await consumer.start()

        messages = await consumer.poll(max_messages=10, timeout_ms=15000)

        assert len(messages) == 1
        assert messages[0].value == b"produced-before-group-joined"

    async def test_latest_silently_skips_message_produced_before_group_joined(
        self,
        bootstrap: str,
        topic: str,
        consumer_factory: list[KafkaTransport],
    ) -> None:
        """Documents the bug this fix removes from the deployed gateway path.

        Same scenario as above, but with the pre-fix
        ``beta-gateway-canary.yaml`` value (``"latest"``): the message
        produced before the group joined is never delivered. This is not a
        delay -- ``poll()`` returns empty even after the full timeout, and
        the message is gone from this group's perspective (its committed
        offset lands after it). ``ModelGatewayForwarderRuntimeConfig`` now
        refuses to resolve a runtime config with this value on either leg
        (see ``tests/unit/runtime/test_gateway_forwarder_runtime.py``); this
        test exercises ``KafkaTransport`` directly to prove the underlying
        broker mechanics the validator exists to prevent.
        """
        group = f"gw-offset-recovery-latest-{uuid4().hex}"
        await _produce_one(
            bootstrap, topic=topic, value=b"produced-before-group-joined"
        )

        consumer = KafkaTransport.from_bootstrap(
            bootstrap, group=group, topics=[topic], auto_offset_reset="latest"
        )
        consumer_factory.append(consumer)
        await consumer.start()

        messages = await consumer.poll(max_messages=10, timeout_ms=5000)

        assert messages == []
