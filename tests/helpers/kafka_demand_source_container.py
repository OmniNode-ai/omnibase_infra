# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A container holding one Kafka event bus, for the boot demand split (OMN-24223).

The demand source takes the container and reads the Kafka bus's own broker
config from it, as the runtime kernel wires it (``ProtocolEventBusSubscriber``).
"""

from __future__ import annotations

from typing import cast

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
    HandlerKafkaTopicDemandSource,
)


class _BusContainer:
    def __init__(self, bus: EventBusKafka) -> None:
        self._bus = bus

    def get_service(self, protocol_type: object) -> EventBusKafka:
        return self._bus


def demand_source_for(bootstrap_servers: str) -> HandlerKafkaTopicDemandSource:
    """The demand source over a not-yet-started Kafka bus at ``bootstrap_servers``."""
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(bootstrap_servers=bootstrap_servers)
    )
    return HandlerKafkaTopicDemandSource(cast("ModelONEXContainer", _BusContainer(bus)))


def demand_source_over_bus(bus: EventBusKafka) -> HandlerKafkaTopicDemandSource:
    """The demand source over a live Kafka bus."""
    return HandlerKafkaTopicDemandSource(cast("ModelONEXContainer", _BusContainer(bus)))
