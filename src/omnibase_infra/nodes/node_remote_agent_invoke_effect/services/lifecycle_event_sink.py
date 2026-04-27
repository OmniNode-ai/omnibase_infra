# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle event publication boundary for remote agent tasks."""

from __future__ import annotations

from typing import Protocol

from omnibase_core.models.delegation.model_agent_task_lifecycle_event import (
    ModelAgentTaskLifecycleEvent,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike


class ProtocolLifecycleEventSink(Protocol):
    """Boundary used by A2A task handling to record lifecycle events."""

    async def write(self, event: ModelAgentTaskLifecycleEvent) -> None:
        """Record a lifecycle event."""


class EventBusLifecycleEventSink:
    """Publish lifecycle events through the configured event bus."""

    def __init__(
        self,
        *,
        event_bus: ProtocolEventBusLike,
        lifecycle_topic: str,
    ) -> None:
        self._event_bus = event_bus
        self._lifecycle_topic = lifecycle_topic

    async def write(self, event: ModelAgentTaskLifecycleEvent) -> None:
        envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=event.correlation_id,
            payload=event.model_dump(mode="json"),
        )
        await self._event_bus.publish(
            self._lifecycle_topic,
            key=str(event.task_id).encode("utf-8"),
            value=envelope.model_dump_json().encode("utf-8"),
        )
