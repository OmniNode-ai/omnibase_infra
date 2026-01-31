# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event bus adapters implementing protocol interfaces.

This module provides adapters that bridge event bus implementations to protocol
interfaces defined in omnibase_spi.

Available Adapters:
    - AdapterProtocolEventPublisherKafka: Kafka implementation of ProtocolEventPublisher

Usage:
    ```python
    from omnibase_infra.event_bus.adapters import AdapterProtocolEventPublisherKafka
    from omnibase_infra.event_bus import EventBusKafka

    bus = EventBusKafka.default()
    await bus.start()

    adapter = AdapterProtocolEventPublisherKafka(bus=bus, service_name="my-service")
    success = await adapter.publish(
        event_type="user.created.v1",
        payload={"user_id": "123"},
    )
    ```
"""

from omnibase_infra.event_bus.adapters.adapter_protocol_event_publisher_kafka import (
    AdapterProtocolEventPublisherKafka,
)

__all__: list[str] = ["AdapterProtocolEventPublisherKafka"]
