# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The in-memory bus's publish path, one batch at a time (OMN-21228).

``EventBusInmemory.publish`` is the one-entry case of ``publish_batch``. A batch
is appended to the bus's history in ONE hold of the bus lock, together with an
optional ``precondition`` that may refuse the whole batch; delivery to
subscribers follows outside the lock. A transactional producer's fence check is
the precondition, so a fence lands before a batch or after it, never in the
middle, and a fenced producer leaves nothing in history and delivers nothing.

The receipt of each entry is minted from the offset assigned under the lock,
BEFORE subscriber fan-out: the record is in this bus's history at that point, so
it is durable *with respect to this bus* regardless of whether a downstream
subscriber then fails. A subscriber's failure is a delivery problem, not a
produce problem, and conflating them would make the in-memory proof diverge from
Kafka (OMN-15861).

Not an ONEX handler (no ``handler_type``/``handler_category``); same arrangement
as ``kafka_permanent_failure``. The bus is typed as the canonical core bus
(OMN-13419): the infra adapter inherits every attribute used here.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from omnibase_core.event_bus.event_bus_inmemory import (
    EventBusInmemory as _CoreEventBusInmemory,
)
from omnibase_core.models.event_bus.model_event_message import (
    ModelEventMessage as _CoreModelEventMessage,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.event_bus.models import (
    ModelEventHeaders,
    ModelEventMessage,
    ModelPublishReceipt,
)

logger = logging.getLogger(__name__)

PublishEntry = tuple[str, bytes | None, bytes, ModelEventHeaders | None]
Subscribers = list[tuple[str, Callable[[_CoreModelEventMessage], Awaitable[None]]]]


async def publish_batch(
    bus: _CoreEventBusInmemory,
    entries: Sequence[PublishEntry],
    precondition: Callable[[], None] | None = None,
) -> tuple[ModelPublishReceipt, ...]:
    """Append every entry under one lock hold, then deliver; one receipt each."""
    if not bus._started:
        first_topic, _, _, first_headers = entries[0]
        raise InfraUnavailableError(
            "Event bus not started. Call start() first.",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INMEMORY,
                operation="publish",
                target_name=f"event_bus.{bus._environment}",
                correlation_id=(
                    first_headers.correlation_id if first_headers else uuid4()
                ),
            ),
            topic=first_topic,
        )

    staged: list[
        tuple[ModelEventMessage, ModelEventHeaders, Subscribers, ModelPublishReceipt]
    ] = []
    async with bus._lock:
        if precondition is not None:
            precondition()
        for topic, key, value, headers in entries:
            if headers is None:
                headers = ModelEventHeaders(
                    source=f"{bus._environment}.{bus._group}",
                    event_type=topic,
                    timestamp=datetime.now(UTC),
                )
            offset = bus._topic_offsets[topic]
            bus._topic_offsets[topic] = offset + 1
            message = ModelEventMessage(
                topic=topic,
                key=key,
                value=value,
                headers=headers,
                offset=str(offset),
                partition=0,
            )
            # OMN-13419: the inherited core deque is typed for the core
            # ModelEventMessage; the adapter intentionally stores the infra-typed
            # message (same structural shape, str schema_version) so consumers
            # receive the infra model. Drops to a re-export once core aligns.
            bus._event_history.append(cast("_CoreModelEventMessage", message))
            receipt = ModelPublishReceipt(
                topic=topic,
                partition=0,
                offset=offset,
                cluster=f"inmemory.{bus._environment}",
                produced_at=datetime.now(UTC),
                transport=EnumInfraTransportType.INMEMORY,
                idempotency_key=headers.idempotency_key,
            )
            staged.append(
                (message, headers, list(bus._subscribers.get(topic, [])), receipt)
            )

    for message, headers, subscribers, _ in staged:
        await _deliver(bus, message, headers, subscribers)
    return tuple(receipt for _, _, _, receipt in staged)


async def _deliver(
    bus: _CoreEventBusInmemory,
    message: ModelEventMessage,
    headers: ModelEventHeaders,
    subscribers: Subscribers,
) -> None:
    """Deliver one appended message; mirror the inherited circuit-breaker policy."""
    topic = message.topic
    for group_id, callback in subscribers:
        failure_key = (topic, group_id)

        async with bus._lock:
            failure_count = bus._subscriber_failures.get(failure_key, 0)

        if failure_count >= bus._max_consecutive_failures:
            logger.warning(
                "Subscriber circuit breaker open - skipping callback",
                extra={
                    "topic": topic,
                    "group_id": group_id,
                    "consecutive_failures": failure_count,
                    "correlation_id": str(headers.correlation_id),
                },
            )
            continue

        try:
            # Callback typed for core ModelEventMessage; infra-typed
            # message is structurally identical (OMN-13419).
            await callback(cast("_CoreModelEventMessage", message))
            async with bus._lock:
                if failure_key in bus._subscriber_failures:
                    del bus._subscriber_failures[failure_key]
        except Exception as e:
            async with bus._lock:
                bus._subscriber_failures[failure_key] = (
                    bus._subscriber_failures.get(failure_key, 0) + 1
                )
                current_failure_count = bus._subscriber_failures[failure_key]
            logger.exception(
                "Subscriber callback failed",
                extra={
                    "topic": topic,
                    "group_id": group_id,
                    "error": str(e),
                    "consecutive_failures": current_failure_count,
                    "correlation_id": str(headers.correlation_id),
                },
            )
