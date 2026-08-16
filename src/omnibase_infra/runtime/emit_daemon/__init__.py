# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Emit Daemon - Event registry and runtime topic-marker infrastructure.

This package provides the generic event registry and the runtime
topic-marker models for the node_emit_daemon_runtime contract.

Components:
- EventRegistry: Maps event types to Kafka topics with metadata injection
- ModelEventRegistration: Configuration model for event type mappings

Note:
    The EventRegistry ships with no default registrations. Consumers must
    register their own event types via ``register()`` or ``register_batch()``.

    The EmitDaemon, EmitClient, and BoundedEventQueue were moved to omniclaude3
    as part of OMN-1944/OMN-1945. Only the shared event registry and the
    runtime topic-marker models remain in omnibase_infra.

    ``NotificationConsumer`` (and its ``ModelNotificationBlocked`` /
    ``ModelNotificationCompleted`` models) was removed under OMN-15970: a
    real-caller search found zero production callers anywhere in this repo —
    only its own docstring example and its own unit tests referenced it, and
    no script/CLI/service_kernel/docker wiring ever instantiated it as a
    running service. It duplicated the canonical, contract-driven
    ``node_slack_alerter_effect`` EFFECT node without ever being wired to
    run. The notification topics themselves (``TOPIC_NOTIFICATION_BLOCKED``
    / ``TOPIC_NOTIFICATION_COMPLETED``) remain live — they are asserted by
    ``omnibase_infra.validation.demo_loop_gate`` and documented as a
    provisioning-continuity topic surface by the ``node_emit_daemon_runtime``
    contract — only the dead consumer class was deleted.

Example Usage:
    ```python
    from omnibase_infra.runtime.emit_daemon import (
        EventRegistry,
        ModelEventRegistration,
    )

    # Create and populate the registry
    registry = EventRegistry(environment="dev")
    registry.register(
        ModelEventRegistration(
            event_type="myapp.submitted",
            topic_template="onex.evt.myapp.submitted.v1",  # onex-topic-allow: pending contract auto-wiring
            partition_key_field="session_id",
            required_fields=("session_id",),
        )
    )
    topic = registry.resolve_topic("myapp.submitted")
    ```
"""

from omnibase_infra.runtime.emit_daemon.event_registry import (
    EventRegistry,
    ModelEventRegistration,
)
from omnibase_infra.runtime.emit_daemon.models import (
    ModelEmitDaemonRuntimeInput,
    ModelEmitDaemonRuntimeOutput,
)
from omnibase_infra.runtime.emit_daemon.topics import (
    PHASE_METRICS_REGISTRATION,
    TOPIC_NOTIFICATION_BLOCKED,
    TOPIC_NOTIFICATION_COMPLETED,
    TOPIC_PHASE_METRICS,
)

__all__: list[str] = [
    "EventRegistry",
    "ModelEmitDaemonRuntimeInput",
    "ModelEmitDaemonRuntimeOutput",
    "ModelEventRegistration",
    "PHASE_METRICS_REGISTRATION",
    "TOPIC_NOTIFICATION_BLOCKED",
    "TOPIC_NOTIFICATION_COMPLETED",
    "TOPIC_PHASE_METRICS",
]
