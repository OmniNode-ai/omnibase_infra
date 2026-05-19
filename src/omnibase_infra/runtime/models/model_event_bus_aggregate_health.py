# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical typed model for the event_bus sub-dict in the :8085/health response.

This module is part of the OMN-9266 response-shape citation work.  The
``event_bus`` field in the ``GET /health`` aggregate response is populated by
``EventBusKafka.health_check()``
(``omnibase_infra/src/omnibase_infra/event_bus/event_bus_kafka.py``,
method ``health_check``, return dict at lines 2200-2210).

See Also:
    :mod:`~omnibase_infra.runtime.models.model_runtime_aggregate_health` —
    outer aggregate model that embeds this as its ``event_bus`` field.

OMN-9266: Cite ServiceHealth aggregate response-shape model for :8085/health.

Example:
    >>> from omnibase_infra.runtime.models.model_event_bus_aggregate_health import (
    ...     ModelEventBusAggregateHealth,
    ... )
    >>>
    >>> raw = {
    ...     "healthy": True,
    ...     "started": True,
    ...     "environment": "prod",
    ...     "bootstrap_servers": "redpanda:9092",
    ...     "circuit_state": "closed",
    ...     "subscriber_count": 374,
    ...     "topic_count": 42,
    ...     "consumer_count": 12,
    ... }
    >>> model = ModelEventBusAggregateHealth.model_validate(raw, strict=False)
    >>> model.circuit_state
    'closed'
    >>> model.subscriber_count
    374
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelEventBusAggregateHealth(BaseModel):
    """Typed model for the ``event_bus`` sub-dict in the :8085/health response.

    Populated by ``EventBusKafka.health_check()`` (return dict, lines 2200-2210
    of ``omnibase_infra/src/omnibase_infra/event_bus/event_bus_kafka.py``).

    The ``circuit_state`` field answers the OMN-9266 open question:
    it comes from the circuit-breaker flag in
    ``EventBusKafka._circuit_breaker_open`` and is serialised as
    ``"open"`` or ``"closed"``.

    Attributes:
        healthy: Whether the Kafka producer is started and its client is open.
        started: Whether ``EventBusKafka.start()`` has been called.
        environment: The ONEX environment string (e.g. ``"prod"``, ``"dev"``).
        bootstrap_servers: Sanitised Kafka bootstrap server address(es).
        circuit_state: Circuit-breaker state — ``"open"`` or ``"closed"``.
        subscriber_count: Total active subscriptions across all topics.
        topic_count: Number of topics that have at least one subscriber.
        consumer_count: Max of individual consumer count and group-consumer count.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        from_attributes=True,
    )

    healthy: bool = Field(
        ...,
        description="Whether the Kafka producer is started and its client is not closed",
    )
    started: bool = Field(
        ...,
        description="Whether EventBusKafka.start() has been called",
    )
    environment: str = Field(
        ...,
        description="ONEX environment string (e.g. 'prod', 'dev')",
    )
    bootstrap_servers: str = Field(
        ...,
        description="Sanitised Kafka bootstrap server address(es)",
    )
    circuit_state: Literal["open", "closed"] = Field(
        ...,
        description=(
            "Circuit-breaker state from EventBusKafka._circuit_breaker_open. "
            "'open' means the breaker has tripped and publish calls may be suppressed; "
            "'closed' is the normal healthy state."
        ),
    )
    subscriber_count: int = Field(
        ...,
        ge=0,
        description="Total active subscriptions across all topics",
    )
    topic_count: int = Field(
        ...,
        ge=0,
        description="Number of topics that have at least one subscriber",
    )
    consumer_count: int = Field(
        ...,
        ge=0,
        description=(
            "Max of individual consumer count and group-consumer count "
            "from EventBusKafka._consumers and _group_consumers"
        ),
    )


__all__: list[str] = ["ModelEventBusAggregateHealth"]
