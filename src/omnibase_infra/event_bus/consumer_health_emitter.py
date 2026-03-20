# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Best-effort consumer health event emitter for Layer 1 pipeline.

Non-blocking, fire-and-forget emitter for consumer health events.
Holds a reference to an existing AIOKafkaProducer (does NOT own the
producer lifecycle). Emission failure is tolerated and logged.

Features:
    - Local rate limiter: max 1 event per fingerprint per 10 seconds
    - Self-metrics: events_emitted, events_dropped, events_rate_limited
    - Feature flag gated: ENABLE_CONSUMER_HEALTH_EMITTER (default off)

Related Tickets:
    - OMN-5516: Create ConsumerHealthEmitter
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.topic_constants import TOPIC_CONSUMER_HEALTH
from omnibase_infra.models.health.enum_consumer_health_event_type import (
    EnumConsumerHealthEventType,
)
from omnibase_infra.models.health.enum_consumer_health_severity import (
    EnumConsumerHealthSeverity,
)
from omnibase_infra.models.health.model_consumer_health_event import (
    ModelConsumerHealthEvent,
)

if TYPE_CHECKING:
    from uuid import UUID

    from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)

# Rate limit: max 1 event per fingerprint per 10 seconds
_RATE_LIMIT_SECONDS = 10.0


class ConsumerHealthEmitter:
    """Best-effort, non-blocking emitter for consumer health events.

    Does NOT own the producer lifecycle -- callers must pass an already-started
    AIOKafkaProducer. Emission failure is tolerated (logged, counted).

    All emission is gated by ENABLE_CONSUMER_HEALTH_EMITTER env var (default off).

    Attributes:
        events_emitted: Count of successfully emitted events.
        events_dropped: Count of events that failed to emit.
        events_rate_limited: Count of events suppressed by rate limiter.
    """

    def __init__(
        self,
        producer: AIOKafkaProducer,
        *,
        topic: str = TOPIC_CONSUMER_HEALTH,
    ) -> None:
        """Initialize the emitter.

        Args:
            producer: An already-started AIOKafkaProducer.
            topic: Topic to emit to (defaults to TOPIC_CONSUMER_HEALTH).
        """
        self._producer = producer
        self._topic = topic
        self._rate_limit_cache: dict[str, float] = {}

        # Self-metrics
        self.events_emitted: int = 0
        self.events_dropped: int = 0
        self.events_rate_limited: int = 0

    @staticmethod
    def is_enabled() -> bool:
        """Check if consumer health emission is enabled via feature flag.

        Returns:
            True if ENABLE_CONSUMER_HEALTH_EMITTER is truthy.
        """
        return (
            os.environ.get(  # ONEX_FLAG_EXEMPT: declared in service-level contract (contracts/services/event_bus.contract.yaml)
                "ENABLE_CONSUMER_HEALTH_EMITTER", ""
            )
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

    def _is_rate_limited(self, fingerprint: str) -> bool:
        """Check if an event with this fingerprint is rate-limited.

        Args:
            fingerprint: Event fingerprint.

        Returns:
            True if the event should be suppressed.
        """
        now = time.monotonic()
        last_emit = self._rate_limit_cache.get(fingerprint, 0.0)
        if now - last_emit < _RATE_LIMIT_SECONDS:
            return True
        self._rate_limit_cache[fingerprint] = now
        return False

    async def emit(self, event: ModelConsumerHealthEvent) -> None:
        """Emit a consumer health event (fire-and-forget).

        Rate-limited per fingerprint. Emission failure is logged but
        does not raise.

        Args:
            event: The consumer health event to emit.
        """
        if not self.is_enabled():
            return

        if self._is_rate_limited(event.fingerprint):
            self.events_rate_limited += 1
            return

        try:
            payload = json.dumps(
                event.model_dump(mode="json"),
            ).encode("utf-8")
            await self._producer.send(self._topic, value=payload)
            self.events_emitted += 1
        except Exception:  # noqa: BLE001 - intentional catch-all for best-effort emission
            self.events_dropped += 1
            logger.debug(
                "Failed to emit consumer health event (fingerprint=%s)",
                event.fingerprint,
                exc_info=True,
            )

    async def emit_event(
        self,
        *,
        consumer_identity: str,
        consumer_group: str,
        topic: str,
        event_type: EnumConsumerHealthEventType,
        severity: EnumConsumerHealthSeverity,
        correlation_id: UUID | None = None,
        rebalance_duration_ms: int | None = None,
        partitions_assigned: int | None = None,
        partitions_revoked: int | None = None,
        error_message: str = "",
        error_type: str = "",
        hostname: str = "",
        service_label: str = "",
    ) -> None:
        """Convenience method to construct and emit in one call.

        Args:
            consumer_identity: Kafka consumer identity string.
            consumer_group: Kafka consumer group ID.
            topic: Primary Kafka topic.
            event_type: Type of health event.
            severity: Severity level.
            correlation_id: Optional correlation ID.
            rebalance_duration_ms: Rebalance duration in ms.
            partitions_assigned: Partitions assigned after rebalance.
            partitions_revoked: Partitions revoked during rebalance.
            error_message: Error message if applicable.
            error_type: Exception type name if applicable.
            hostname: Machine hostname.
            service_label: Service display label.
        """
        event = ModelConsumerHealthEvent.create(
            consumer_identity=consumer_identity,
            consumer_group=consumer_group,
            topic=topic,
            event_type=event_type,
            severity=severity,
            correlation_id=correlation_id,
            rebalance_duration_ms=rebalance_duration_ms,
            partitions_assigned=partitions_assigned,
            partitions_revoked=partitions_revoked,
            error_message=error_message,
            error_type=error_type,
            hostname=hostname,
            service_label=service_label,
        )
        await self.emit(event)


__all__ = ["ConsumerHealthEmitter"]
