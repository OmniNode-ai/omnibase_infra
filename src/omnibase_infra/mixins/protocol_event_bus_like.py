# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Minimal event bus protocol for components that only need publish capability.

This module defines a narrow interface following the Interface Segregation Principle.
Components that only need to publish events can depend on ProtocolEventBusLike
instead of the full ProtocolEventBus, reducing coupling and simplifying testing.

Example Usage:
    ```python
    from omnibase_infra.mixins import ProtocolEventBusLike

    class TimeoutEmitter:
        def __init__(self, event_bus: ProtocolEventBusLike):
            self._event_bus = event_bus

        async def emit_timeout(self, event: MyEvent, topic: str) -> None:
            await self._event_bus.publish_envelope(event, topic)
    ```

Related:
    - KafkaEventBus: Full implementation with circuit breaker, retry, etc.
    - InMemoryEventBus: In-memory implementation for testing
    - ProtocolEventBus: Full event bus protocol (omnibase_spi)
"""

from typing import Protocol


class ProtocolEventBusLike(Protocol):
    """Minimal protocol for event bus operations needed by timeout emitter.

    This is a narrow interface following the Interface Segregation Principle.
    Components that only need to publish events can depend on this instead
    of the full ProtocolEventBus, reducing coupling and simplifying testing.

    The protocol uses `object` for the envelope type instead of `Any` to satisfy
    ONEX "no Any types" guidelines while maintaining flexibility for any envelope
    type that can be serialized.

    Implementors:
        - KafkaEventBus: Production Kafka-based implementation
        - InMemoryEventBus: Testing and development implementation

    Thread Safety:
        Implementations should be thread-safe and support concurrent calls
        to publish_envelope from multiple coroutines.
    """

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Publish an event envelope to a topic.

        Args:
            envelope: The event envelope to publish. Must be serializable
                (typically a Pydantic model with model_dump_json support).
            topic: The topic name to publish to. Must be a valid Kafka topic
                name or equivalent for the underlying implementation.

        Raises:
            InfraConnectionError: If connection to the message broker fails.
            InfraTimeoutError: If the publish operation times out.
            InfraUnavailableError: If circuit breaker is open.

        Example:
            ```python
            await event_bus.publish_envelope(
                envelope=my_event,
                topic="dev.myapp.events.v1",
            )
            ```
        """
        ...


__all__ = ["ProtocolEventBusLike"]
