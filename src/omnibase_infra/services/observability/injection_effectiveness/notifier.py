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
    >>> from omnibase_infra.services.observability.injection_effectiveness.notifier import (
    ...     EffectivenessInvalidationNotifier,
    ... )
    >>>
    >>> producer = AIOKafkaProducer(bootstrap_servers="localhost:9092")
    >>> await producer.start()
    >>> notifier = EffectivenessInvalidationNotifier(producer)
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

from omnibase_infra.services.observability.injection_effectiveness.models.model_invalidation_event import (
    ModelEffectivenessInvalidationEvent,
)

logger = logging.getLogger(__name__)

TOPIC_EFFECTIVENESS_INVALIDATION: str = (
    "onex.evt.omnibase-infra.effectiveness-data-changed.v1"
)
"""Kafka topic for effectiveness data invalidation events.

Consumers (dashboard WebSocket servers, API caches) subscribe to this topic
to know when to refresh effectiveness data.
"""


class EffectivenessInvalidationNotifier:
    """Publishes invalidation events when effectiveness data changes.

    Best-effort notification: failures are logged at WARNING level but
    never propagate. The caller's write operation has already succeeded
    and must not be rolled back due to a notification failure.

    Attributes:
        _producer: Kafka producer for publishing invalidation events.
        _topic: Target Kafka topic for invalidation events.

    Example:
        >>> notifier = EffectivenessInvalidationNotifier(producer)
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

        This method is fire-and-forget: it logs failures but never raises.

        Args:
            tables_affected: Names of the tables that were updated.
            rows_written: Total rows written in this batch.
            source: Whether data came from kafka_consumer or batch_compute.
            correlation_id: Optional correlation ID for tracing.
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
    "EffectivenessInvalidationNotifier",
    "TOPIC_EFFECTIVENESS_INVALIDATION",
]
