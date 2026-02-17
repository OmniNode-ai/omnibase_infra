# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Invalidation notifier for effectiveness measurement updates.

Publishes Kafka events when new measurement data is written to effectiveness
tables, enabling downstream consumers (dashboards, WebSocket servers) to
refresh their cached data.

Design Decisions:
    - Fire-and-forget: Notification failures are logged but never raise.
      Measurement persistence is the primary concern; notifications are
      best-effort.
    - Single topic: All invalidation events go to one topic. Consumers
      filter by tables_affected if they only care about specific tables.
    - JSON serialization: Events are serialized as JSON for broad
      consumer compatibility.

Related Tickets:
    - OMN-2303: Activate effectiveness consumer and populate measurement tables

Example:
    >>> from aiokafka import AIOKafkaProducer
    >>> from omnibase_infra.services.observability.injection_effectiveness.service_effectiveness_invalidation_notifier import (
    ...     ServiceEffectivenessInvalidationNotifier,
    ... )
    >>>
    >>> producer = AIOKafkaProducer(bootstrap_servers="localhost:9092")
    >>> await producer.start()
    >>> notifier = ServiceEffectivenessInvalidationNotifier(producer)
    >>>
    >>> await notifier.notify(
    ...     tables_affected=("injection_effectiveness", "pattern_hit_rates"),
    ...     rows_written=42,
    ...     source="kafka_consumer",
    ...     correlation_id=some_uuid,
    ... )
"""

from __future__ import annotations

import json
import logging
from typing import Literal
from uuid import UUID, uuid4

from aiokafka import AIOKafkaProducer

from omnibase_infra.event_bus.topic_constants import (
    TOPIC_EFFECTIVENESS_INVALIDATION,
)
from omnibase_infra.services.observability.injection_effectiveness.models.model_invalidation_event import (
    ModelEffectivenessInvalidationEvent,
)

logger = logging.getLogger(__name__)


class ServiceEffectivenessInvalidationNotifier:
    """Publishes invalidation events when effectiveness data changes.

    Best-effort notification: failures are logged at WARNING level but
    never propagate. The caller's write operation has already succeeded
    and must not be rolled back due to a notification failure.

    Attributes:
        _producer: Kafka producer for publishing invalidation events.
        _topic: Target Kafka topic for invalidation events.

    Example:
        >>> notifier = ServiceEffectivenessInvalidationNotifier(producer)
        >>> await notifier.notify(
        ...     tables_affected=("injection_effectiveness",),
        ...     rows_written=10,
        ...     source="batch_compute",
        ... )
    """

    def __init__(
        self,
        producer: AIOKafkaProducer,
        topic: str = TOPIC_EFFECTIVENESS_INVALIDATION,
    ) -> None:
        """Initialize the notifier with a Kafka producer.

        Args:
            producer: An already-started AIOKafkaProducer instance.
                Lifecycle is managed externally.
            topic: Kafka topic for invalidation events. Defaults to
                the standard effectiveness invalidation topic.
        """
        self._producer = producer
        self._topic = topic

    async def notify(
        self,
        *,
        tables_affected: tuple[str, ...],
        rows_written: int,
        source: Literal["kafka_consumer", "batch_compute"],
        correlation_id: UUID | None = None,
    ) -> None:
        """Publish an invalidation event for effectiveness data changes.

        This method is fire-and-forget: Kafka send failures are caught and
        logged at WARNING level but never propagated. The caller's write
        has already succeeded and must not be affected by notification
        failure.

        No-ops silently when ``rows_written`` is zero or negative.

        Args:
            tables_affected: Names of the effectiveness tables that were
                updated (e.g., ``("injection_effectiveness",)``).
            rows_written: Total number of rows written in this batch.
                Values <= 0 cause an early return with no event published.
            source: Origin of the data write -- ``"kafka_consumer"`` for
                real-time pipeline writes, ``"batch_compute"`` for
                backfill writes.
            correlation_id: Optional correlation ID for tracing. A new
                UUID is generated if not provided.

        Raises:
            This method never raises. All exceptions from Kafka
            serialization or publishing are caught internally.
        """
        if rows_written <= 0:
            return

        effective_correlation_id = correlation_id or uuid4()

        event = ModelEffectivenessInvalidationEvent(
            correlation_id=effective_correlation_id,
            tables_affected=tables_affected,
            rows_written=rows_written,
            source=source,
        )

        try:
            payload = json.dumps(
                event.model_dump(mode="json"),
            ).encode("utf-8")

            await self._producer.send_and_wait(
                self._topic,
                value=payload,
                key=str(effective_correlation_id).encode("utf-8"),
            )

            logger.debug(
                "Published effectiveness invalidation event",
                extra={
                    "topic": self._topic,
                    "tables_affected": tables_affected,
                    "rows_written": rows_written,
                    "source": source,
                    "correlation_id": str(effective_correlation_id),
                },
            )

        except Exception:
            # Best-effort: log but never raise
            logger.warning(
                "Failed to publish effectiveness invalidation event",
                extra={
                    "topic": self._topic,
                    "tables_affected": tables_affected,
                    "rows_written": rows_written,
                    "source": source,
                    "correlation_id": str(effective_correlation_id),
                },
                exc_info=True,
            )


__all__ = [
    "ServiceEffectivenessInvalidationNotifier",
]
