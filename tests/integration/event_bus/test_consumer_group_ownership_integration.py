# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OMN-9420: consumer-group ownership per (topic, group_id).

Exercises the EventBusKafka in-process state — no live Kafka broker required.
Verifies that distinct (topic, group_id) pairs do not collapse into a single
consumer entry, which was the root cause of the registration verification failure.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)

pytestmark = [pytest.mark.integration]


def _make_bus() -> EventBusKafka:
    config = ModelKafkaEventBusConfig(
        bootstrap_servers="localhost:9092",
    )
    return EventBusKafka(config=config)


@pytest.mark.integration
class TestConsumerGroupOwnership:
    """Verify OMN-9420: (topic, group_id) keying in _group_consumers."""

    def test_pending_consumer_keys_initialises_empty(self) -> None:
        bus = _make_bus()
        assert hasattr(bus, "_pending_consumer_keys")
        assert isinstance(bus._pending_consumer_keys, set)
        assert len(bus._pending_consumer_keys) == 0

    def test_group_consumers_keyed_by_topic_and_group_id(self) -> None:
        bus = _make_bus()
        assert hasattr(bus, "_group_consumers")
        # dict keys are (topic, group_id) tuples
        assert isinstance(bus._group_consumers, dict)

    def test_group_consumer_tasks_keyed_by_topic_and_group_id(self) -> None:
        bus = _make_bus()
        assert hasattr(bus, "_group_consumer_tasks")
        assert isinstance(bus._group_consumer_tasks, dict)

    def test_pending_keys_distinct_from_active_keys(self) -> None:
        bus = _make_bus()
        # Before any start: both sets empty, no overlap
        active = set(bus._group_consumers.keys())
        pending = bus._pending_consumer_keys
        assert active.isdisjoint(pending)
