# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-configuration for backend registry using onex.backends entry points.

Discovers installed backends via entry points, probes them for health,
and registers the best available backend for each protocol in the
container's service registry.

When omnibase_core's auto_configure_registry() is available (Part 1 merged),
this module delegates to it. Until then, it provides equivalent functionality
using the same entry point group and probe model.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points

from omnibase_infra.backends.backend_probe import (
    probe_kafka,
    probe_postgres,
)
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult

logger = logging.getLogger(__name__)

# Probe functions keyed by entry point name
_PROBE_REGISTRY: dict[str, object] = {
    "event_bus_kafka": probe_kafka,
    "state_postgres": probe_postgres,
}


def discover_backends() -> list[ModelProbeResult]:
    """Discover and probe all installed onex.backends entry points.

    Returns:
        List of probe results, one per discovered backend.
    """
    results: list[ModelProbeResult] = []
    backends = entry_points(group="onex.backends")

    for ep in backends:
        probe_fn = _PROBE_REGISTRY.get(ep.name)
        if probe_fn is not None and callable(probe_fn):
            result = probe_fn()
            results.append(result)
        else:
            results.append(
                ModelProbeResult(
                    state=EnumProbeState.DISCOVERED,
                    reason=f"No probe registered for backend '{ep.name}'",
                    backend_label=ep.name,
                )
            )

    return results


def select_event_bus(
    *,
    kafka_bootstrap_servers: str | None = None,
    environment: str = "local",
    consumer_group: str = "onex-runtime",
    circuit_breaker_threshold: int = 5,
) -> object:
    """Select the best available event bus based on backend probes.

    Uses the onex.backends entry point probes to determine whether Kafka
    is available. Falls back to EventBusInmemory when Kafka is not healthy.

    This replaces the inline if/else bus creation in service_kernel.py.

    Args:
        kafka_bootstrap_servers: Kafka broker addresses.
        environment: Runtime environment identifier.
        consumer_group: Consumer group for the bus.
        circuit_breaker_threshold: Circuit breaker threshold.

    Returns:
        An event bus instance (EventBusKafka or EventBusInmemory).
    """
    # Check environment override first
    bus_type_override = os.getenv("ONEX_EVENT_BUS_TYPE", "").lower()
    if bus_type_override == "inmemory":
        logger.info("Using EventBusInmemory (ONEX_EVENT_BUS_TYPE override)")
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

        return EventBusInmemory(
            environment=environment,
            group=consumer_group,
        )

    # Probe Kafka backend
    kafka_result = probe_kafka(bootstrap_servers=kafka_bootstrap_servers)

    if kafka_result.state in (
        EnumProbeState.HEALTHY,
        EnumProbeState.AUTHORITATIVE,
    ):
        logger.info(
            "Kafka probe: %s — using EventBusKafka",
            kafka_result.reason,
        )
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        kafka_config = ModelKafkaEventBusConfig(
            bootstrap_servers=kafka_bootstrap_servers or "",
            environment=environment,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )
        return EventBusKafka(config=kafka_config)

    if kafka_result.state == EnumProbeState.REACHABLE and kafka_bootstrap_servers:
        # Reachable but not healthy — still try Kafka if explicitly configured
        logger.warning(
            "Kafka probe: %s — attempting EventBusKafka despite probe result",
            kafka_result.reason,
        )
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        kafka_config = ModelKafkaEventBusConfig(
            bootstrap_servers=kafka_bootstrap_servers,
            environment=environment,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )
        return EventBusKafka(config=kafka_config)

    # Fallback to in-memory
    logger.info(
        "Kafka probe: %s — falling back to EventBusInmemory",
        kafka_result.reason,
    )
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory

    return EventBusInmemory(
        environment=environment,
        group=consumer_group,
    )
