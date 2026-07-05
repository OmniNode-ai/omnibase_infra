# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-runner health assessment (OMN-13942) -- the COMPUTE node's classification output."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_health_state import (
    EnumRunnerFleetHealthState,
)


class ModelRunnerHealthAssessment(BaseModel):
    """Classified health state for a single runner."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ..., description="Runner name (matches ModelRunnerFleetRunnerFact.name)."
    )
    state: EnumRunnerFleetHealthState = Field(
        ..., description="Classified health state."
    )
    detail: str = Field(
        default="", description="Explanation of why this state was chosen."
    )


__all__ = ["ModelRunnerHealthAssessment"]
