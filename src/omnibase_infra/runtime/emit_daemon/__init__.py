# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Hook Event Daemon - Event emission infrastructure for OmniClaude hooks.

This package provides the daemon and supporting infrastructure for emitting
events from OmniClaude hooks to Kafka topics.

Components:
- EmitDaemon: Unix socket server for persistent Kafka event emission
- EmitClient: Client for emitting events via the daemon
- EventRegistry: Maps event types to Kafka topics with metadata injection
- ModelEventRegistration: Configuration model for event type mappings
- ModelEmitDaemonConfig: Configuration model for the daemon
- BoundedEventQueue: In-memory queue with disk spool overflow
- ModelQueuedEvent: Event model for queued events

Example Usage:
    ```python
    from omnibase_infra.runtime.emit_daemon import (
        EmitClient,
        EmitDaemon,
        EventRegistry,
        ModelEmitDaemonConfig,
        ModelEventRegistration,
        emit_event,
    )

    # Client usage (recommended for hooks)
    async with EmitClient() as client:
        event_id = await client.emit("prompt.submitted", {"prompt_id": "abc123"})

    # Or use convenience function
    event_id = await emit_event("prompt.submitted", {"prompt_id": "abc123"})

    # Daemon usage (for running the server)
    config = ModelEmitDaemonConfig(
        kafka_bootstrap_servers="kafka:9092",
    )
    daemon = EmitDaemon(config)
    await daemon.start()
    await daemon.run_until_shutdown()

    # Or use the registry directly
    registry = EventRegistry(environment="dev")
    topic = registry.resolve_topic("prompt.submitted")
    ```
"""

from omnibase_infra.runtime.emit_daemon.cli import main as cli_main
from omnibase_infra.runtime.emit_daemon.client import (
    EmitClient,
    EmitClientError,
    emit_event,
    emit_event_with_fallback,
)
from omnibase_infra.runtime.emit_daemon.config import ModelEmitDaemonConfig
from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon
from omnibase_infra.runtime.emit_daemon.event_registry import (
    EventRegistry,
    ModelEventRegistration,
)
from omnibase_infra.runtime.emit_daemon.model_daemon_request import (
    ModelDaemonEmitRequest,
    ModelDaemonPingRequest,
    parse_daemon_request,
)
from omnibase_infra.runtime.emit_daemon.model_daemon_response import (
    ModelDaemonErrorResponse,
    ModelDaemonPingResponse,
    ModelDaemonQueuedResponse,
    parse_daemon_response,
)
from omnibase_infra.runtime.emit_daemon.queue import (
    BoundedEventQueue,
    ModelQueuedEvent,
)

__all__: list[str] = [
    "BoundedEventQueue",
    "EmitClient",
    "EmitClientError",
    "EmitDaemon",
    "EventRegistry",
    "ModelDaemonEmitRequest",
    "ModelDaemonErrorResponse",
    "ModelDaemonPingRequest",
    "ModelDaemonPingResponse",
    "ModelDaemonQueuedResponse",
    "ModelEmitDaemonConfig",
    "ModelEventRegistration",
    "ModelQueuedEvent",
    "cli_main",
    "emit_event",
    "emit_event_with_fallback",
    "parse_daemon_request",
    "parse_daemon_response",
]
