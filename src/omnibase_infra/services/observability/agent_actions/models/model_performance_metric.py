# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance Metric Model.

This module defines the model for router performance metrics consumed
from Kafka. Performance metrics capture timing and throughput data
for the agent routing system.

Design Decisions:
    - extra="allow": Phase 1 flexibility - required fields typed, extras preserved
    - raw_payload: Optional field to preserve complete payload for schema tightening
    - created_at: Required for TTL cleanup job (Phase 2)

Idempotency:
    Table: router_performance_metrics
    Unique Key: id (UUID)
    Conflict Action: DO NOTHING (append-only time-series)

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> metric = ModelPerformanceMetric(
    ...     id=uuid4(),
    ...     metric_name="routing_latency_ms",
    ...     metric_value=45.2,
    ...     created_at=datetime.now(UTC),
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelPerformanceMetric(BaseModel):
    """Router performance metric model.

    Represents a single performance metric data point for the routing
    system. Used for monitoring latency, throughput, and resource usage.

    Attributes:
        id: Unique identifier for this metric (idempotency key).
        metric_name: Name of the metric being recorded.
        metric_value: Numeric value of the metric.
        created_at: Timestamp when the metric was recorded (TTL key).
        correlation_id: Optional request correlation ID for trace-specific metrics.
        unit: Optional unit of measurement (ms, bytes, count, etc.).
        agent_name: Optional agent name if metric is agent-specific.
        labels: Optional key-value labels for metric dimensionality.
        metadata: Optional additional metadata about the metric.
        raw_payload: Optional complete raw payload for Phase 2 schema tightening.

    Example:
        >>> metric = ModelPerformanceMetric(
        ...     id=uuid4(),
        ...     metric_name="agent_selection_time_ms",
        ...     metric_value=12.5,
        ...     created_at=datetime.now(UTC),
        ...     unit="ms",
        ...     agent_name="polymorphic-agent",
        ...     labels={"operation": "route", "status": "success"},
        ... )
    """

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
    )

    # ---- Required Fields ----
    id: UUID = Field(
        ...,
        description="Unique identifier for this metric (idempotency key).",
    )
    metric_name: str = Field(  # ONEX_EXCLUDE: entity_reference - external payload
        ..., description="Name of the metric being recorded."
    )
    metric_value: float = Field(
        ...,
        description="Numeric value of the metric.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the metric was recorded (TTL key).",
    )

    # ---- Optional Fields ----
    correlation_id: UUID | None = Field(
        default=None,
        description="Request correlation ID for trace-specific metrics.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit of measurement (ms, bytes, count, etc.).",
    )
    agent_name: str | None = Field(
        default=None,
        description="Agent name if metric is agent-specific.",
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Key-value labels for metric dimensionality.",
    )
    metadata: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional metadata about the metric.",
    )
    raw_payload: dict[str, JsonType] | None = Field(
        default=None,
        description="Complete raw payload for Phase 2 schema tightening.",
    )

    def __str__(self) -> str:
        """Return concise string representation for logging.

        Includes key identifying fields but excludes metadata and raw_payload.
        """
        id_short = str(self.id)[:8]
        unit_part = f" {self.unit}" if self.unit else ""
        agent_part = f", agent={self.agent_name}" if self.agent_name else ""
        return (
            f"PerformanceMetric(id={id_short}, "
            f"{self.metric_name}={self.metric_value}{unit_part}{agent_part})"
        )


__all__ = ["ModelPerformanceMetric"]
