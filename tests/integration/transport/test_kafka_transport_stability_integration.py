# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live-broker readback for the Kafka transport face (OMN-14756, S3, plan (g)/S3).

Stronger than the conformance suite's restart-inference: this asserts the group's
**committed offset value** advances on ``commit`` (read back via the admin API's
``list_consumer_group_offsets``), and that ``nack`` replays the message plus later
same-partition offsets on the live consumer. Uses a throwaway per-test topic
(``transport.s3.integration.*`` — deliberately NOT an ``onex.*`` delegation topic)
so it never disturbs real traffic.

Intended target: a ``.201`` broker with topic-create headroom
(``ONEX_TRANSPORT_KAFKA_BOOTSTRAP=<broker>``). Skipped when no broker is reachable.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.event_bus.kafka_transport import KafkaTransport

from ._kafka_env import (
    committed_offset,
    delete_topic,
    kafka_available,
    recreate_topic,
    transport_bootstrap,
    transport_topic_prefix,
)

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


async def _drain(
    consumer: KafkaTransport, *, max_polls: int = 20
) -> list[ModelTransportMessage]:
    collected: list[ModelTransportMessage] = []
    for _ in range(max_polls):
        batch = await consumer.poll(max_messages=64, timeout_ms=200)
        if not batch:
            if collected:
                break
            continue
        collected.extend(batch)
    return collected


@pytest.fixture
def broker_bootstrap() -> str:
    return transport_bootstrap()


@pytest.fixture
async def throwaway_topic(broker_bootstrap: str) -> AsyncIterator[str]:
    topic = f"{transport_topic_prefix()}.{uuid.uuid4().hex[:8]}.v1"
    await recreate_topic(broker_bootstrap, topic, partitions=1)
    try:
        yield topic
    finally:
        # Best-effort cleanup: drop the throwaway topic.
        await delete_topic(broker_bootstrap, topic)


class TestKafkaTransportStabilityReadback:
    """poll -> commit advances the committed offset; nack replays (live broker)."""

    async def test_commit_advances_committed_offset(
        self, broker_bootstrap: str, throwaway_topic: str
    ) -> None:
        topic = throwaway_topic
        group = f"s3-commit-{uuid.uuid4().hex[:8]}"

        producer = KafkaTransport.from_bootstrap(broker_bootstrap)
        await producer.start()
        try:
            for i in range(3):
                await producer.send(topic, key=b"k", value=str(i).encode(), headers={})
        finally:
            await producer.close()

        consumer = KafkaTransport.from_bootstrap(
            broker_bootstrap, group=group, topics=[topic]
        )
        await consumer.start()
        try:
            batch = await _drain(consumer)
            assert len(batch) == 3, f"expected 3 messages, got {len(batch)}"
            base = min(m.offset for m in batch)

            # No commit yet -> group has no committed offset for this partition.
            before = await committed_offset(broker_bootstrap, group, topic, 0)
            assert before in (None, -1), f"expected no committed offset, got {before}"

            middle = next(m for m in batch if m.offset == base + 1)
            await consumer.commit(middle)

            # Committed offset is the NEXT offset to fetch: committing message at
            # base+1 records base+2, proving all <= base+1 are durable.
            after = await committed_offset(broker_bootstrap, group, topic, 0)
            assert after == base + 2, (
                f"commit at offset {base + 1} must advance committed offset to "
                f"{base + 2}, got {after}"
            )
        finally:
            await consumer.close()

        # Restart on the same group: only base+2 (uncommitted) is redelivered.
        restart = KafkaTransport.from_bootstrap(
            broker_bootstrap, group=group, topics=[topic]
        )
        await restart.start()
        try:
            redelivered = await _drain(restart)
            offsets = sorted(m.offset for m in redelivered)
            assert offsets == [base + 2], (
                f"expected only offset {base + 2} redelivered after commit, got "
                f"{offsets}"
            )
        finally:
            await restart.close()

    async def test_nack_replays_message_and_later(
        self, broker_bootstrap: str, throwaway_topic: str
    ) -> None:
        topic = throwaway_topic
        group = f"s3-nack-{uuid.uuid4().hex[:8]}"

        producer = KafkaTransport.from_bootstrap(broker_bootstrap)
        await producer.start()
        try:
            for i in range(3):
                await producer.send(topic, key=b"k", value=str(i).encode(), headers={})
        finally:
            await producer.close()

        consumer = KafkaTransport.from_bootstrap(
            broker_bootstrap, group=group, topics=[topic]
        )
        await consumer.start()
        try:
            batch = await _drain(consumer)
            assert len(batch) == 3
            base = min(m.offset for m in batch)
            target = next(m for m in batch if m.offset == base + 1)

            await consumer.nack(target)
            replayed = await _drain(consumer)
            offsets = sorted(m.offset for m in replayed)
            assert offsets == [base + 1, base + 2], (
                f"nack must replay the message and later same-partition offsets, "
                f"got {offsets}"
            )
        finally:
            await consumer.close()
