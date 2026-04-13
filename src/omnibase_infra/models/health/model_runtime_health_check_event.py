# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime health check event model.

Emitted by ServiceRuntimeHealthMonitor every check interval.

Schema aligns with ``onex.evt.omnibase-infra.runtime-health-check.v1``.

.. versionadded:: 0.39.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.model_runtime_health_dimension import (
    ModelRuntimeHealthDimension,
)


class ModelRuntimeHealthCheckEvent(BaseModel):
    """Runtime health check snapshot emitted by ServiceRuntimeHealthMonitor.

    Consumers (dashboards, alerting) can subscribe to
    ``onex.evt.omnibase-infra.runtime-health-check.v1`` to receive these events.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Correlation ID for tracing")
    timestamp: datetime = Field(..., description="UTC timestamp of the check")
    status: Literal["HEALTHY", "DEGRADED", "CRITICAL"] = Field(
        ..., description="Aggregate health status derived from worst dimension"
    )
    dimensions: tuple[ModelRuntimeHealthDimension, ...] = Field(
        default_factory=tuple,
        description="Per-dimension health breakdown",
    )
    contract_count: int = Field(
        default=0,
        description="Number of contracts discovered by the auto-wiring engine",
    )
    discovery_error_count: int = Field(
        default=0,
        description="Number of errors encountered during contract discovery",
    )
    consumer_group_count: int = Field(
        default=0,
        description="Total consumer groups visible to the Kafka admin client",
    )
    empty_consumer_group_count: int = Field(
        default=0,
        description="Consumer groups that are Empty (no active members)",
    )
    subscribe_topic_count: int = Field(
        default=0,
        description="Topics declared as subscribe targets across all contracts",
    )
    uncovered_topic_count: int = Field(
        default=0,
        description="Subscribe topics with no matching non-empty consumer group",
    )


__all__: list[str] = ["ModelRuntimeHealthCheckEvent"]
