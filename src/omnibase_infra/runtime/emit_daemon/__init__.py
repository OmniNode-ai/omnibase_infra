# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Emit Daemon - Event infrastructure for OmniClaude hooks.

This package provides the event registry and notification infrastructure
for OmniClaude hook events.

Components:
- EventRegistry: Maps event types to Kafka topics with metadata injection
- ModelEventRegistration: Configuration model for event type mappings
- NotificationConsumer: Consumes notification events and routes to Slack
- ModelNotificationBlocked: Event model for blocked notifications
- ModelNotificationCompleted: Event model for completion notifications

Note:
    The EmitDaemon, EmitClient, and BoundedEventQueue were moved to omniclaude3
    as part of OMN-1944/OMN-1945. Only the shared event registry, notification
    consumer, and notification models remain in omnibase_infra.

Example Usage:
    ```python
    from omnibase_infra.runtime.emit_daemon import (
        EventRegistry,
        ModelEventRegistration,
        NotificationConsumer,
    )

    # Use the registry directly
    registry = EventRegistry(environment="dev")
    topic = registry.resolve_topic("prompt.submitted")

    # Notification consumer usage
    consumer = NotificationConsumer(event_bus=kafka_event_bus)
    await consumer.start()
    ```
"""

from omnibase_infra.runtime.emit_daemon.event_registry import (
    EventRegistry,
    ModelEventRegistration,
)
from omnibase_infra.runtime.emit_daemon.models import (
    ModelNotificationBlocked,
    ModelNotificationCompleted,
)
from omnibase_infra.runtime.emit_daemon.notification_consumer import (
    TOPIC_NOTIFICATION_BLOCKED,
    TOPIC_NOTIFICATION_COMPLETED,
    NotificationConsumer,
)

__all__: list[str] = [
    "EventRegistry",
    "ModelEventRegistration",
    "ModelNotificationBlocked",
    "ModelNotificationCompleted",
    "NotificationConsumer",
    "TOPIC_NOTIFICATION_BLOCKED",
    "TOPIC_NOTIFICATION_COMPLETED",
]
