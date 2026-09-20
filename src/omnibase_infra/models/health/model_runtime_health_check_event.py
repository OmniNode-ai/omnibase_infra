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
    projection_count: int = Field(
        default=0,
        description=(
            "Contract-declared projections in scope for the liveness dimensions "
            "(OMN-16994)"
        ),
    )
    unattached_projection_count: int = Field(
        default=0,
        description=(
            "Declared projections with no attached consumer — they persist "
            "nothing while every lag-based check reads green (OMN-16994)"
        ),
    )
    dlq_saturated_projection_count: int = Field(
        default=0,
        description=(
            "Attached projections routing 100% of consumed events to a DLQ or "
            "quarantine sink over the observation window (OMN-16994)"
        ),
    )
    lane: str | None = Field(
        default=None,
        description=(
            "OMN-18769. The runtime lane this verdict is ABOUT -- compose-dev, "
            "onex-lab, onex-lab-k3s -- read from the ONEX_RUNTIME_LANE "
            "environment variable at emit time.\n"
            "\n"
            "Nullable, and deliberately so. A runtime whose deployment does "
            "not set the variable genuinely does not know which lane it is, "
            "and a default would be a guess: this event is consumed by a "
            "per-lane projection, and a guessed lane puts one cluster's "
            "verdict onto another lane's row. A consumer that cannot read a "
            "lane here must DROP the event rather than attribute it, which is "
            "what node_projection_lab_lane_health does.\n"
            "\n"
            "It is not a required field because that would make every "
            "already-deployed runtime fail to construct its own health event "
            "-- turning an observability improvement into an outage."
        ),
    )
    nonwriting_projection_count: int = Field(
        default=0,
        description=(
            "Attached projections whose in-process dispatch is a deliberate "
            "no-op (standalone-runner shape): offsets commit and nothing is "
            "written here, and the two counts above read 0 through it because "
            "the topic IS attached and nothing raises (OMN-17448)"
        ),
    )


__all__: list[str] = ["ModelRuntimeHealthCheckEvent"]
