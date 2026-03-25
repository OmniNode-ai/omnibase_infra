# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test that EventBusKafka does NOT implement consumption counting.

Bug history:
- OMN-6441: Quick-fixed by adding MixinConsumptionCounter to EventBusKafka.
- OMN-6515: Corrected fix — consumption counting belongs on
  EventBusSubcontractWiring, which tracks messages successfully processed
  by handlers. EventBusKafka only handles emission counting.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.observability.wiring_health import MixinConsumptionCounter


@pytest.mark.unit
class TestEventBusKafkaNoConsumptionCounter:
    """Verify EventBusKafka does NOT provide get_consumption_counts().

    Consumption counting belongs on EventBusSubcontractWiring (OMN-6515).
    """

    def test_does_not_have_get_consumption_counts_method(self) -> None:
        """EventBusKafka must NOT have get_consumption_counts() method."""
        assert not hasattr(EventBusKafka, "get_consumption_counts"), (
            "EventBusKafka must NOT implement get_consumption_counts(). "
            "Consumption counting belongs on EventBusSubcontractWiring (OMN-6515)."
        )

    def test_mixin_not_in_mro(self) -> None:
        """MixinConsumptionCounter must NOT appear in EventBusKafka MRO."""
        assert MixinConsumptionCounter not in EventBusKafka.__mro__, (
            "EventBusKafka must NOT inherit MixinConsumptionCounter (OMN-6515)."
        )
