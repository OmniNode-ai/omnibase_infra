# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MixinConsumptionCounter."""

from __future__ import annotations

import asyncio

import pytest

from omnibase_infra.event_bus.topic_constants import WIRING_HEALTH_MONITORED_TOPICS
from omnibase_infra.observability.wiring_health import MixinConsumptionCounter

pytestmark = pytest.mark.unit


class _ConsumptionCounterTestable(MixinConsumptionCounter):
    """Testable class that uses the consumption counter mixin."""

    def __init__(self) -> None:
        self._init_consumption_counter()


class TestMixinConsumptionCounter:
    """Tests for MixinConsumptionCounter."""

    @pytest.fixture
    def counter(self) -> _ConsumptionCounterTestable:
        """Create a testable consumption counter."""
        return _ConsumptionCounterTestable()

    @pytest.mark.asyncio
    async def test_init_creates_empty_counts(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """Initialization should create empty consumption counts."""
        counts = counter.get_consumption_counts()
        assert counts == {}

    @pytest.mark.asyncio
    async def test_record_consumption_for_monitored_topic(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """Recording consumption for monitored topic should increment count."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]

        await counter._record_consumption(topic)
        counts = counter.get_consumption_counts()

        assert counts[topic] == 1

    @pytest.mark.asyncio
    async def test_record_consumption_increments_count(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """Multiple consumptions should increment count correctly."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]

        await counter._record_consumption(topic)
        await counter._record_consumption(topic)
        await counter._record_consumption(topic)

        counts = counter.get_consumption_counts()
        assert counts[topic] == 3

    @pytest.mark.asyncio
    async def test_record_consumption_for_unmonitored_topic(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """Recording consumption for unmonitored topic should be ignored."""
        unmonitored_topic = "some.unmonitored.topic.v1"

        # Verify it's not monitored
        assert unmonitored_topic not in counter.get_consumption_monitored_topics()

        await counter._record_consumption(unmonitored_topic)
        counts = counter.get_consumption_counts()

        # Should not be in counts
        assert unmonitored_topic not in counts
        assert counts == {}

    @pytest.mark.asyncio
    async def test_get_consumption_counts_returns_copy(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """get_consumption_counts should return a copy, not internal state."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]
        await counter._record_consumption(topic)

        counts1 = counter.get_consumption_counts()
        counts1[topic] = 999  # Modify the copy

        counts2 = counter.get_consumption_counts()
        assert counts2[topic] == 1  # Original unchanged

    @pytest.mark.asyncio
    async def test_reset_consumption_counts(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """Reset should clear counts and return old values."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]
        await counter._record_consumption(topic)
        await counter._record_consumption(topic)

        old_counts = await counter.reset_consumption_counts()

        assert old_counts[topic] == 2
        assert counter.get_consumption_counts() == {}

    @pytest.mark.asyncio
    async def test_get_consumption_monitored_topics(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """get_consumption_monitored_topics should return frozen set."""
        topics = counter.get_consumption_monitored_topics()

        assert isinstance(topics, frozenset)
        assert len(topics) > 0
        for topic in WIRING_HEALTH_MONITORED_TOPICS:
            assert topic in topics

    @pytest.mark.asyncio
    async def test_concurrent_consumptions(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """Concurrent consumptions should be handled correctly."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]
        num_consumptions = 100

        async def consume() -> None:
            await counter._record_consumption(topic)

        # Run many concurrent consumptions
        await asyncio.gather(*[consume() for _ in range(num_consumptions)])

        counts = counter.get_consumption_counts()
        assert counts[topic] == num_consumptions

    @pytest.mark.asyncio
    async def test_multiple_monitored_topics(
        self, counter: _ConsumptionCounterTestable
    ) -> None:
        """Should track consumptions for multiple monitored topics separately."""
        topics = list(WIRING_HEALTH_MONITORED_TOPICS)[:2]
        if len(topics) < 2:
            pytest.skip("Need at least 2 monitored topics")

        await counter._record_consumption(topics[0])
        await counter._record_consumption(topics[0])
        await counter._record_consumption(topics[1])

        counts = counter.get_consumption_counts()
        assert counts[topics[0]] == 2
        assert counts[topics[1]] == 1
