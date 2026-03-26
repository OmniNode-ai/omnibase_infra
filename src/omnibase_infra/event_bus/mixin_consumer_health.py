# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mixin providing consumer health emission for standalone consumers.

Standalone Kafka consumers (those NOT managed by EventBusKafka) use this
mixin to emit health events and handle restart commands.

Usage:
    class MyConsumer(MixinConsumerHealth):
        async def start(self):
            await self._init_health_emitter(producer, consumer_identity="my-consumer", ...)
            ...

        async def _handle_restart(self) -> None:
            # Implement consumer-specific restart logic
            ...

Related Tickets:
    - OMN-5519: Create MixinConsumerHealth for standalone consumers
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.consumer_health_emitter import ConsumerHealthEmitter
from omnibase_infra.models.health.enum_consumer_health_event_type import (
    EnumConsumerHealthEventType,
)
from omnibase_infra.models.health.enum_consumer_health_severity import (
    EnumConsumerHealthSeverity,
)

if TYPE_CHECKING:
    from uuid import UUID

    from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)


class MixinConsumerHealth:
    """Mixin providing consumer health emission for standalone consumers.

    Provides:
        - Health event emission via ConsumerHealthEmitter
        - Convenience methods for common event types
        - Feature flag gating via ENABLE_CONSUMER_HEALTH_EMITTER

    Subclasses should call ``_init_health_emitter()`` during startup and
    use ``_emit_health_event()`` to emit events.
    """

    _health_emitter: ConsumerHealthEmitter | None

    def _init_health_emitter(
        self,
        producer: AIOKafkaProducer,
        *,
        consumer_identity: str,
        consumer_group: str,
        topic: str,
        service_label: str = "",
        hostname: str = "",
    ) -> None:
        """Initialize the health emitter.

        Args:
            producer: An already-started AIOKafkaProducer.
            consumer_identity: Kafka consumer identity string.
            consumer_group: Consumer group ID.
            topic: Primary topic the consumer subscribes to.
            service_label: Service display label.
            hostname: Machine hostname.
        """
        self._health_emitter = ConsumerHealthEmitter(producer)
        self._health_consumer_identity = consumer_identity
        self._health_consumer_group = consumer_group
        self._health_topic = topic
        self._health_service_label = service_label
        self._health_hostname = hostname

    async def _emit_health_event(
        self,
        event_type: EnumConsumerHealthEventType,
        severity: EnumConsumerHealthSeverity,
        *,
        correlation_id: UUID | None = None,
        error_message: str = "",
        error_type: str = "",
        rebalance_duration_ms: int | None = None,
        partitions_assigned: int | None = None,
        partitions_revoked: int | None = None,
    ) -> None:
        """Emit a consumer health event if emitter is initialized and enabled.

        Args:
            event_type: Type of health event.
            severity: Severity level.
            correlation_id: Optional correlation ID.
            error_message: Error message if applicable.
            error_type: Exception type name if applicable.
            rebalance_duration_ms: Rebalance duration in ms.
            partitions_assigned: Partitions assigned after rebalance.
            partitions_revoked: Partitions revoked during rebalance.
        """
        emitter = getattr(self, "_health_emitter", None)
        if emitter is None:
            return

        try:
            await emitter.emit_event(
                consumer_identity=self._health_consumer_identity,
                consumer_group=self._health_consumer_group,
                topic=self._health_topic,
                event_type=event_type,
                severity=severity,
                correlation_id=correlation_id,
                error_message=error_message,
                error_type=error_type,
                rebalance_duration_ms=rebalance_duration_ms,
                partitions_assigned=partitions_assigned,
                partitions_revoked=partitions_revoked,
                hostname=self._health_hostname,
                service_label=self._health_service_label,
            )
        except Exception:  # noqa: BLE001 - best-effort emission
            logger.debug(
                "Failed to emit health event for %s",
                getattr(self, "_health_consumer_identity", "unknown"),
                exc_info=True,
            )

    async def _emit_heartbeat_failure(
        self,
        error_message: str = "",
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Convenience: emit heartbeat failure event."""
        await self._emit_health_event(
            EnumConsumerHealthEventType.HEARTBEAT_FAILURE,
            EnumConsumerHealthSeverity.ERROR,
            error_message=error_message,
            correlation_id=correlation_id,
        )

    async def _emit_session_timeout(
        self,
        error_message: str = "",
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        """Convenience: emit session timeout event."""
        await self._emit_health_event(
            EnumConsumerHealthEventType.SESSION_TIMEOUT,
            EnumConsumerHealthSeverity.CRITICAL,
            error_message=error_message,
            correlation_id=correlation_id,
        )

    async def _emit_consumer_started(self) -> None:
        """Convenience: emit consumer started event."""
        await self._emit_health_event(
            EnumConsumerHealthEventType.CONSUMER_STARTED,
            EnumConsumerHealthSeverity.INFO,
        )

    async def _emit_consumer_stopped(self) -> None:
        """Convenience: emit consumer stopped event."""
        await self._emit_health_event(
            EnumConsumerHealthEventType.CONSUMER_STOPPED,
            EnumConsumerHealthSeverity.WARNING,
        )


__all__ = ["MixinConsumerHealth"]
