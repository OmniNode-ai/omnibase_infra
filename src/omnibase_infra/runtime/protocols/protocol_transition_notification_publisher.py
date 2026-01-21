# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for transition notification publishing services.

This module defines the ProtocolTransitionNotificationPublisher interface for
services that publish state transition notifications to event buses. The outbox
implementation uses this protocol to decouple from specific event bus implementations.

Architecture Context:
    In the ONEX registration domain:
    - Projectors (F1) update projections and write notifications to outbox
    - TransitionNotificationOutbox processes outbox entries
    - Publishers (implementing this protocol) deliver to event buses
    - Orchestrators and other consumers react to transition notifications

Concurrency Safety:
    Implementations must be coroutine-safe for concurrent async publishing.
    Multiple coroutines may invoke publish() concurrently. Use asyncio.Lock
    for shared mutable state (coroutine-safe, not thread-safe).

Example Usage:
    ```python
    from omnibase_infra.runtime.protocols import ProtocolTransitionNotificationPublisher
    from omnibase_infra.runtime.models import ModelStateTransitionNotification

    class KafkaTransitionPublisher:
        '''Concrete implementation publishing to Kafka.'''

        async def publish(
            self,
            notification: ModelStateTransitionNotification,
        ) -> None:
            '''Publish notification to Kafka topic.'''
            key = notification.to_topic_key()
            await self._kafka_producer.send(
                topic="registration.transitions",
                key=key.encode("utf-8"),
                value=notification.model_dump_json().encode("utf-8"),
            )

    # Protocol conformance check via duck typing (per ONEX conventions)
    publisher = KafkaTransitionPublisher()
    assert hasattr(publisher, 'publish') and callable(publisher.publish)
    ```

Related Tickets:
    - OMN-1139: TransitionNotificationOutbox implementation

.. versionadded:: 0.8.0
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.runtime.models.model_state_transition_notification import (
    ModelStateTransitionNotification,
)

__all__ = [
    "ProtocolTransitionNotificationPublisher",
]


@runtime_checkable
class ProtocolTransitionNotificationPublisher(Protocol):
    """Protocol for state transition notification publishing services.

    Defines the interface for publishing state transition notifications to
    event buses. The TransitionNotificationOutbox uses this protocol to
    deliver notifications that were stored during projection writes.

    Concurrency Safety:
        Implementations must be coroutine-safe for concurrent async publishing.
        Multiple coroutines may invoke publish() concurrently.

    Error Handling:
        Implementations should raise OnexError subclasses on failure:
        - InfraConnectionError: Event bus unavailable
        - InfraTimeoutError: Publish operation timed out
        - ProtocolConfigurationError: Invalid topic/serialization config

    Example Implementation:
        ```python
        class KafkaTransitionPublisher:
            def __init__(self, producer: KafkaProducer, topic: str):
                self._producer = producer
                self._topic = topic

            async def publish(
                self,
                notification: ModelStateTransitionNotification,
            ) -> None:
                key = notification.to_topic_key().encode("utf-8")
                value = notification.model_dump_json().encode("utf-8")
                await self._producer.send(self._topic, key=key, value=value)
        ```
    """

    async def publish(
        self,
        notification: ModelStateTransitionNotification,
    ) -> None:
        """Publish a state transition notification to the event bus.

        Publishes the notification to the configured topic/channel. The
        notification key should be derived from aggregate_type and aggregate_id
        to ensure proper partitioning.

        Args:
            notification: The state transition notification to publish.
                Contains all transition details including aggregate info,
                state change, and correlation ID.

        Raises:
            InfraConnectionError: If event bus is unavailable
            InfraTimeoutError: If publish operation times out
            OnexError: For serialization or configuration errors

        Example:
            ```python
            notification = ModelStateTransitionNotification(
                notification_id=uuid4(),
                aggregate_type="registration",
                aggregate_id=node_id,
                from_state="pending_registration",
                to_state="awaiting_ack",
                event_type="NodeRegistrationAccepted",
                event_id=event.message_id,
                correlation_id=correlation_id,
                occurred_at=datetime.now(UTC),
            )
            await publisher.publish(notification)
            ```

        Implementation Notes:
            - Key should be notification.to_topic_key() for partitioning
            - Value should be JSON-serialized notification
            - Include correlation_id in publish context for tracing
        """
        ...
