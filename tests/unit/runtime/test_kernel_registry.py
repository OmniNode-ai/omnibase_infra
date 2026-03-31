# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for kernel registry auto-configuration (OMN-7076).

Tests that the kernel resolves the event bus from the registry based on
backend probes, not inline if/else creation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.backends.auto_configure import (
    select_event_bus,
)
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult

pytestmark = pytest.mark.unit


class TestKernelRegistryResolution:
    """Test that the kernel resolves event bus from registry."""

    def test_registry_resolves_inmemory_when_kafka_unavailable(self) -> None:
        """When Kafka is unreachable, kernel auto-falls back to in-memory bus."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason="TCP connect to localhost:59999 failed",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("ONEX_EVENT_BUS_TYPE", None)
            bus = select_event_bus(
                kafka_bootstrap_servers=None,
                environment="test",
                consumer_group="test-group",
            )
            assert type(bus).__name__ == "EventBusInmemory"

    def test_registry_resolves_kafka_when_healthy(self) -> None:
        """When Kafka is healthy, kernel selects EventBusKafka."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason="Kafka healthy with 5 topics, brokers match config",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("ONEX_EVENT_BUS_TYPE", None)
            bus = select_event_bus(
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
            )
            assert type(bus).__name__ == "EventBusKafka"

    def test_env_override_forces_inmemory(self) -> None:
        """ONEX_EVENT_BUS_TYPE=inmemory forces in-memory regardless of probe."""
        with patch.dict("os.environ", {"ONEX_EVENT_BUS_TYPE": "inmemory"}):
            bus = select_event_bus(
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
            )
            assert type(bus).__name__ == "EventBusInmemory"

    def test_reachable_with_explicit_servers_uses_kafka(self) -> None:
        """When Kafka is REACHABLE and bootstrap_servers set, still try Kafka."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason="TCP reachable but topic list failed",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("ONEX_EVENT_BUS_TYPE", None)
            bus = select_event_bus(
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
            )
            assert type(bus).__name__ == "EventBusKafka"
