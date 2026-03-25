# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Interface-surface regression tests for wiring health count sources.

These tests verify that the concrete classes used as emission and consumption
sources in production have the expected method interfaces. They are regression
guards, not full protocol conformance proofs — they check method presence,
mixin inheritance, and the absence of methods on the wrong class.

Root cause guard for OMN-6515: EventBusKafka was passed as consumption_source
but lacked get_consumption_counts(), causing AttributeError every 60s.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestEmissionSourceInterface:
    """Verify EventBusKafka has the emission counting interface."""

    def test_event_bus_kafka_inherits_emission_mixin(self) -> None:
        """EventBusKafka must inherit MixinEmissionCounter."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.observability.wiring_health.mixin_emission_counter import (
            MixinEmissionCounter,
        )

        assert issubclass(EventBusKafka, MixinEmissionCounter), (
            "EventBusKafka must inherit MixinEmissionCounter "
            "to satisfy ProtocolEmissionCountSource"
        )

    def test_event_bus_kafka_has_get_emission_counts(self) -> None:
        """EventBusKafka must have callable get_emission_counts()."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        method = getattr(EventBusKafka, "get_emission_counts", None)
        assert method is not None and callable(method), (
            "EventBusKafka.get_emission_counts must exist and be callable"
        )


class TestConsumptionSourceInterface:
    """Verify EventBusSubcontractWiring has the consumption counting interface."""

    def test_subcontract_wiring_inherits_consumption_mixin(self) -> None:
        """EventBusSubcontractWiring must inherit MixinConsumptionCounter."""
        from omnibase_infra.observability.wiring_health.mixin_consumption_counter import (
            MixinConsumptionCounter,
        )
        from omnibase_infra.runtime.event_bus_subcontract_wiring import (
            EventBusSubcontractWiring,
        )

        assert issubclass(EventBusSubcontractWiring, MixinConsumptionCounter), (
            "EventBusSubcontractWiring must inherit MixinConsumptionCounter "
            "to satisfy ProtocolConsumptionCountSource"
        )

    def test_subcontract_wiring_has_get_consumption_counts(self) -> None:
        """EventBusSubcontractWiring must have callable get_consumption_counts()."""
        from omnibase_infra.runtime.event_bus_subcontract_wiring import (
            EventBusSubcontractWiring,
        )

        method = getattr(EventBusSubcontractWiring, "get_consumption_counts", None)
        assert method is not None and callable(method), (
            "EventBusSubcontractWiring.get_consumption_counts must exist and be callable"
        )


class TestWrongFixGuard:
    """Guard against the wrong fix: adding consumption counting to EventBusKafka."""

    def test_event_bus_kafka_does_not_have_consumption_method(self) -> None:
        """EventBusKafka must NOT have get_consumption_counts().

        This guards against the panic fix of adding get_consumption_counts()
        to EventBusKafka instead of wiring the correct source object.
        Consumption counting belongs on EventBusSubcontractWiring because
        it tracks messages successfully processed by handlers, not messages
        published to Kafka.
        """
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

        assert not hasattr(EventBusKafka, "get_consumption_counts"), (
            "EventBusKafka must NOT implement get_consumption_counts(). "
            "Consumption counting belongs on EventBusSubcontractWiring. "
            "If you're seeing this, you may be applying the wrong fix — "
            "wire EventBusSubcontractWiring as the consumption_source instead."
        )

    def test_event_bus_kafka_does_not_inherit_consumption_mixin(self) -> None:
        """EventBusKafka must NOT inherit MixinConsumptionCounter."""
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.observability.wiring_health.mixin_consumption_counter import (
            MixinConsumptionCounter,
        )

        assert not issubclass(EventBusKafka, MixinConsumptionCounter), (
            "EventBusKafka must NOT inherit MixinConsumptionCounter. "
            "If this fails, someone added the wrong mixin to EventBusKafka."
        )
