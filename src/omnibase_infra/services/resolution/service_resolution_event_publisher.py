# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Publisher service for resolution decision audit events.

Publishes ``ModelResolutionEventLocal`` instances to the
``onex.evt.platform.resolution-decided.v1`` topic via the event bus,
recording every tiered dependency resolution decision for audit, replay,
and intelligence.

Design Decisions:
    - **Bus-agnostic**: Accepts any object satisfying ``ProtocolEventBusLike``
      (both ``EventBusKafka`` and ``EventBusInmemory`` work).
    - **Fire-and-forget**: Publish failures are logged at WARNING level and
      never raised. The resolution decision has already been made; audit
      publishing must not block or fail the resolution path.
    - **JSON serialization**: Events are serialized via Pydantic
      ``model_dump(mode="json")`` for broad consumer compatibility.
    - **Service layer**: Named ``Service*`` (not ``Handler*``) to comply
      with ARCH-002 (handlers must not publish to the event bus).

Architecture:
    This service sits at the service layer, called by orchestrators or
    the tiered resolution service after a resolution decision is made.
    It wraps the event bus publish call with:
    - Structured logging with correlation tracking
    - JSON serialization of the resolution event model
    - Error isolation (publish failures never propagate)

Related:
    - OMN-2895: Resolution Event Ledger (Phase 6 of OMN-2897 epic)
    - TOPIC_RESOLUTION_DECIDED: ``onex.evt.platform.resolution-decided.v1``
    - ProtocolEventBusLike: Minimal event bus protocol
    - ModelResolutionEventLocal: Local event model (until core PR #575 merges)
"""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

from omnibase_infra.event_bus.topic_constants import TOPIC_RESOLUTION_DECIDED
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.services.resolution.model_resolution_event_local import (
    ModelResolutionEventLocal,
)

logger = logging.getLogger(__name__)


class ServiceResolutionEventPublisher:
    """Publishes resolution decision events to the event bus.

    This service accepts ``ModelResolutionEventLocal`` instances and
    publishes them as JSON to ``onex.evt.platform.resolution-decided.v1``.
    It is fire-and-forget: publish failures are caught and logged but
    never raised.

    Attributes:
        _bus: The event bus to publish to (Kafka or in-memory).
        _topic: Target topic for resolution events.

    Example:
        >>> from omnibase_infra.event_bus import EventBusInmemory
        >>> from omnibase_infra.services.resolution import (
        ...     ServiceResolutionEventPublisher,
        ...     ModelResolutionEventLocal,
        ... )
        >>>
        >>> bus = EventBusInmemory(environment="test", group="test-group")
        >>> await bus.start()
        >>> publisher = ServiceResolutionEventPublisher(bus)
        >>>
        >>> event = ModelResolutionEventLocal(
        ...     dependency_capability="database.relational",
        ...     success=True,
        ... )
        >>> await publisher.publish(event)
    """

    def __init__(
        self,
        bus: ProtocolEventBusLike,
        topic: str = TOPIC_RESOLUTION_DECIDED,
    ) -> None:
        """Initialize the publisher with an event bus.

        Args:
            bus: An event bus instance satisfying ``ProtocolEventBusLike``.
                Both ``EventBusKafka`` and ``EventBusInmemory`` are supported.
                Lifecycle is managed externally.
            topic: Target topic for resolution events. Defaults to
                ``TOPIC_RESOLUTION_DECIDED``.
        """
        self._bus = bus
        self._topic = topic

    @property
    def topic(self) -> str:
        """Get the target topic name.

        Returns:
            The topic this publisher emits to.
        """
        return self._topic

    async def publish(
        self,
        event: ModelResolutionEventLocal,
        *,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Publish a resolution decision event to the event bus.

        Serializes the event to JSON and publishes to the configured topic.
        This method is fire-and-forget: publish failures are caught and
        logged at WARNING level but never propagated.

        Args:
            event: The resolution event to publish.
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, the event's ``event_id`` is used as the
                partition key for ordering.

        Returns:
            True if the event was published successfully, False otherwise.

        Raises:
            asyncio.CancelledError: Propagated to allow proper async task
                cancellation. All other exceptions are caught internally.
        """
        effective_correlation_id = correlation_id or uuid4()
        key_bytes = str(event.event_id).encode("utf-8")

        try:
            payload_dict = event.to_publishable_dict()
            value_bytes = json.dumps(payload_dict).encode("utf-8")

            # Prefer publish_envelope if available, but use raw publish
            # since our payload is already JSON bytes.
            await self._bus.publish(
                topic=self._topic,
                key=key_bytes,
                value=value_bytes,
            )

            logger.debug(
                "Published resolution event",
                extra={
                    "topic": self._topic,
                    "event_id": str(event.event_id),
                    "dependency_capability": event.dependency_capability,
                    "success": event.success,
                    "failure_code": event.failure_code,
                    "tier_count": len(event.tier_progression),
                    "correlation_id": str(effective_correlation_id),
                },
            )

            return True

        except Exception:
            # Best-effort: log but never raise
            logger.warning(
                "Failed to publish resolution event",
                extra={
                    "topic": self._topic,
                    "event_id": str(event.event_id),
                    "dependency_capability": event.dependency_capability,
                    "success": event.success,
                    "correlation_id": str(effective_correlation_id),
                },
                exc_info=True,
            )

            return False

    async def publish_dict(
        self,
        event_data: dict[str, object],
        *,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Publish a resolution event from a raw dictionary.

        Convenience method for callers that already have the event data
        as a dictionary (e.g., from cross-service communication). The
        dictionary is validated against ``ModelResolutionEventLocal``
        before publishing.

        Args:
            event_data: Dictionary containing resolution event fields.
                Must be valid according to ``ModelResolutionEventLocal``.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if published successfully, False otherwise.
        """
        try:
            event = ModelResolutionEventLocal.model_validate(event_data)
        except Exception:
            logger.warning(
                "Invalid resolution event data, cannot publish",
                extra={
                    "topic": self._topic,
                    "correlation_id": str(correlation_id or "none"),
                },
                exc_info=True,
            )
            return False

        return await self.publish(event, correlation_id=correlation_id)


__all__: list[str] = ["ServiceResolutionEventPublisher"]
