# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Terminal event for the runner-fleet-maintain workflow (OMN-13942, Increment 1).

Carries the health report ONLY -- Increment 1 performs no fleet mutation, so
there is nothing else to report. Increment 2 (design-only) would extend this
workflow to additionally route ``verdict.recommended_actions`` through a
grant-gated recovery EFFECT before reaching this terminal event.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_verdict import (
    ModelRunnerFleetHealthVerdict,
)


class ModelRunnerFleetMaintainCompletedEvent(BaseModel):
    """Terminal event: one runner-fleet-maintain tick completed."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    verdict: ModelRunnerFleetHealthVerdict = Field(
        ..., description="Health report for this tick. No mutation occurred."
    )
