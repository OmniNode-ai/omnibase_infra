# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test that EventBusKafka implements ProtocolConsumptionCountSource.

Bug (OMN-6441): WiringHealthChecker calls get_consumption_counts() on
EventBusKafka, which raises AttributeError because MixinConsumptionCounter
was not mixed in.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.observability.wiring_health import MixinConsumptionCounter


@pytest.mark.unit
class TestEventBusKafkaConsumptionCounter:
    """Verify EventBusKafka provides get_consumption_counts()."""

    def test_has_get_consumption_counts_method(self) -> None:
        """EventBusKafka must have get_consumption_counts() method."""
        assert hasattr(EventBusKafka, "get_consumption_counts")
        assert callable(EventBusKafka.get_consumption_counts)

    def test_mixin_in_mro(self) -> None:
        """MixinConsumptionCounter must appear in EventBusKafka MRO."""
        assert MixinConsumptionCounter in EventBusKafka.__mro__

    def test_satisfies_protocol_structurally(self) -> None:
        """EventBusKafka must have the method required by ProtocolConsumptionCountSource."""
        bus = EventBusKafka()
        # Structural protocol check: method exists and returns correct type
        assert hasattr(bus, "get_consumption_counts")
        result = bus.get_consumption_counts()
        assert isinstance(result, dict)

    def test_returns_dict(self) -> None:
        """get_consumption_counts() must return a dict[str, int]."""
        bus = EventBusKafka()
        counts = bus.get_consumption_counts()
        assert isinstance(counts, dict)
