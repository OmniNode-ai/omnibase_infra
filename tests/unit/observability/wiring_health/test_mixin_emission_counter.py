# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MixinEmissionCounter."""

from __future__ import annotations

import asyncio

import pytest

from omnibase_infra.event_bus.topic_constants import WIRING_HEALTH_MONITORED_TOPICS
from omnibase_infra.observability.wiring_health import MixinEmissionCounter


class _EmissionCounterTestable(MixinEmissionCounter):
    """Testable class that uses the emission counter mixin."""

    def __init__(self) -> None:
        self._init_emission_counter()


class TestMixinEmissionCounter:
    """Tests for MixinEmissionCounter."""

    @pytest.fixture
    def counter(self) -> _EmissionCounterTestable:
        """Create a testable emission counter."""
        return _EmissionCounterTestable()

    @pytest.mark.asyncio
    async def test_init_creates_empty_counts(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """Initialization should create empty emission counts."""
        counts = counter.get_emission_counts()
        assert counts == {}

    @pytest.mark.asyncio
    async def test_record_emission_for_monitored_topic(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """Recording emission for monitored topic should increment count."""
        # Use first monitored topic
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]

        await counter._record_emission(topic)
        counts = counter.get_emission_counts()

        assert counts[topic] == 1

    @pytest.mark.asyncio
    async def test_record_emission_increments_count(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """Multiple emissions should increment count correctly."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]

        await counter._record_emission(topic)
        await counter._record_emission(topic)
        await counter._record_emission(topic)

        counts = counter.get_emission_counts()
        assert counts[topic] == 3

    @pytest.mark.asyncio
    async def test_record_emission_for_unmonitored_topic(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """Recording emission for unmonitored topic should be ignored."""
        unmonitored_topic = "some.unmonitored.topic.v1"

        # Verify it's not monitored
        assert unmonitored_topic not in counter.get_monitored_topics()

        await counter._record_emission(unmonitored_topic)
        counts = counter.get_emission_counts()

        # Should not be in counts
        assert unmonitored_topic not in counts
        assert counts == {}

    @pytest.mark.asyncio
    async def test_get_emission_counts_returns_copy(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """get_emission_counts should return a copy, not internal state."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]
        await counter._record_emission(topic)

        counts1 = counter.get_emission_counts()
        counts1[topic] = 999  # Modify the copy

        counts2 = counter.get_emission_counts()
        assert counts2[topic] == 1  # Original unchanged

    @pytest.mark.asyncio
    async def test_reset_emission_counts(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """Reset should clear counts and return old values."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]
        await counter._record_emission(topic)
        await counter._record_emission(topic)

        old_counts = await counter.reset_emission_counts()

        assert old_counts[topic] == 2
        assert counter.get_emission_counts() == {}

    @pytest.mark.asyncio
    async def test_get_monitored_topics(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """get_monitored_topics should return frozen set of monitored topics."""
        topics = counter.get_monitored_topics()

        assert isinstance(topics, frozenset)
        assert len(topics) > 0
        # Should contain the wiring health monitored topics
        for topic in WIRING_HEALTH_MONITORED_TOPICS:
            assert topic in topics

    @pytest.mark.asyncio
    async def test_concurrent_emissions(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """Concurrent emissions should be handled correctly."""
        topic = WIRING_HEALTH_MONITORED_TOPICS[0]
        num_emissions = 100

        async def emit() -> None:
            await counter._record_emission(topic)

        # Run many concurrent emissions
        await asyncio.gather(*[emit() for _ in range(num_emissions)])

        counts = counter.get_emission_counts()
        assert counts[topic] == num_emissions

    @pytest.mark.asyncio
    async def test_multiple_monitored_topics(
        self, counter: _EmissionCounterTestable
    ) -> None:
        """Should track emissions for multiple monitored topics separately."""
        topics = list(WIRING_HEALTH_MONITORED_TOPICS)[:2]
        if len(topics) < 2:
            pytest.skip("Need at least 2 monitored topics")

        await counter._record_emission(topics[0])
        await counter._record_emission(topics[0])
        await counter._record_emission(topics[1])

        counts = counter.get_emission_counts()
        assert counts[topics[0]] == 2
        assert counts[topics[1]] == 1
