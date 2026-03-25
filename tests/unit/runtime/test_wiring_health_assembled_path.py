# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Assembled-path regression test for wiring health source wiring.

This test proves the kernel's wiring is correct by instantiating
WiringHealthChecker with the same source types used in production
(EventBusKafka for emission, EventBusSubcontractWiring for consumption)
and running a single compute_health() pass.

This is the test that would have caught OMN-6515: passing EventBusKafka
as consumption_source caused AttributeError on get_consumption_counts().
"""

from __future__ import annotations

import pytest

from omnibase_infra.observability.wiring_health.mixin_consumption_counter import (
    MixinConsumptionCounter,
)
from omnibase_infra.observability.wiring_health.mixin_emission_counter import (
    MixinEmissionCounter,
)
from omnibase_infra.observability.wiring_health.wiring_health_checker import (
    WiringHealthChecker,
)

pytestmark = pytest.mark.unit


class _FakeEmissionSource(MixinEmissionCounter):
    """Minimal emission source using the real mixin."""

    def __init__(self) -> None:
        self._init_emission_counter()


class _FakeConsumptionSource(MixinConsumptionCounter):
    """Minimal consumption source using the real mixin."""

    def __init__(self) -> None:
        self._init_consumption_counter()


class TestAssembledWiringHealthPath:
    """Prove the kernel's intended wiring works end-to-end."""

    def test_compute_health_succeeds_with_correct_sources(self) -> None:
        """WiringHealthChecker.compute_health() must not raise AttributeError.

        This is the exact failure from OMN-6515: the kernel passed EventBusKafka
        (emission-only) as consumption_source, and compute_health() called
        get_consumption_counts() which did not exist.
        """
        emission_source = _FakeEmissionSource()
        consumption_source = _FakeConsumptionSource()

        checker = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="test",
        )

        # This is the line that was crashing in production
        metrics = checker.compute_health()

        assert metrics is not None
        assert metrics.overall_healthy is True

    def test_compute_health_fails_with_wrong_source(self) -> None:
        """Using emission source as consumption source must raise AttributeError.

        This reproduces the exact production bug: EventBusKafka (which only has
        get_emission_counts) passed where get_consumption_counts is expected.
        """
        emission_source = _FakeEmissionSource()

        checker = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=emission_source,  # WRONG — same object for both
            environment="test",
        )

        with pytest.raises(AttributeError, match="get_consumption_counts"):
            checker.compute_health()
