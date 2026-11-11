# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The setting reaches a real ``AIOKafkaConsumer`` for the host state topics (OMN-22430).

No broker is contacted: aiokafka records ``isolation_level`` when the consumer
is constructed and uses it on the first fetch. The unit tests patch the
consumer class; this one builds the real class through the event bus.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.event_bus import ModelReadCommittedTopics

_UNLISTED_TOPIC = "onex.evt.omnimarket.something-else.v1"


@pytest.mark.asyncio
async def test_real_consumer_of_each_declared_topic_reads_committed() -> None:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    )
    for topic in ModelReadCommittedTopics.load().topics:
        consumer = bus._build_consumer(topic, "grp", "inst", "earliest", group_id="grp")
        try:
            assert consumer._isolation_level == "read_committed"
        finally:
            await consumer.stop()


@pytest.mark.asyncio
async def test_real_consumer_of_an_unlisted_topic_keeps_the_default() -> None:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    )
    consumer = bus._build_consumer(
        _UNLISTED_TOPIC, "grp", "inst", "earliest", group_id="grp"
    )
    try:
        assert consumer._isolation_level == "read_uncommitted"
    finally:
        await consumer.stop()
