# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-fleet health verdict (OMN-13942) -- the COMPUTE node's terminal output.

Read-only: this model carries the classified per-runner states, fleet
aggregates, and recommended actions. Nothing consumes ``recommended_actions``
to mutate the fleet in Increment 1 -- they are recorded/surfaced only.

OMN-14228 Slice A adds ``github_source_ok``/``docker_source_ok``/
``buildx_determinate`` so a future remediation gate can fail CLOSED on
indeterminate health instead of silently treating a source outage as a
verified HEALTHY fleet.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_recommended_action import (
    ModelRecommendedAction,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_health_assessment import (
    ModelRunnerHealthAssessment,
)


class ModelRunnerFleetHealthVerdict(BaseModel):
    """Classified fleet health verdict: per-runner states + fleet aggregates."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    evaluated_at: datetime = Field(..., description="When classification ran.")
    assessments: tuple[ModelRunnerHealthAssessment, ...] = Field(
        default_factory=tuple, description="Per-runner classified health state."
    )
    expected_count: int = Field(..., ge=0, description="Configured runner count.")
    observed_count: int = Field(..., ge=0, description="Runners actually observed.")
    online_count: int = Field(..., ge=0, description="Runners GitHub reports online.")
    offline_count: int = Field(..., ge=0, description="Runners GitHub reports offline.")
    busy_count: int = Field(..., ge=0, description="Online runners executing a job.")
    idle_count: int = Field(
        ..., ge=0, description="Online runners with no job in flight."
    )
    saturation_ratio: float = Field(
        ..., ge=0.0, description="busy_count / online_count (0.0 if no runners online)."
    )
    crash_looping_count: int = Field(
        ..., ge=0, description="Runners classified CRASH_LOOPING."
    )
    listener_zombie_count: int = Field(
        ..., ge=0, description="Runners classified LISTENER_ZOMBIE."
    )
    wedged_count: int = Field(..., ge=0, description="Runners classified WEDGED.")
    buildx_unavailable: bool = Field(
        default=False,
        description="Whether the OMN-13932 buildx probe reported unavailable.",
    )
    buildx_determinate: bool = Field(
        default=True,
        description=(
            "False when the buildx probe could not determine availability "
            "(snapshot.buildx_available was None). `buildx_unavailable=False` "
            "with `buildx_determinate=False` means UNKNOWN, not confirmed "
            "available -- consumers must not collapse the two."
        ),
    )
    codeload_throttle_signal_count: int = Field(
        default=0, ge=0, description="Codeload-throttle failure signatures observed."
    )
    recommended_actions: tuple[ModelRecommendedAction, ...] = Field(
        default_factory=tuple,
        description="Recorded/surfaced remediation recommendations. NEVER executed here.",
    )
    source_errors: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Upstream snapshot source errors, passed through.",
    )
    github_source_ok: bool = Field(
        default=True,
        description=(
            "Passed through from ModelRunnerFleetSnapshot.github_source_ok. "
            "False means the GitHub API source failed fleet-wide for this "
            "tick -- online/offline/busy classifications below are unreliable."
        ),
    )
    docker_source_ok: bool = Field(
        default=True,
        description=(
            "Passed through from ModelRunnerFleetSnapshot.docker_source_ok. "
            "False means the SSH/Docker inspection source failed fleet-wide "
            "for this tick -- CRASH_LOOPING/LISTENER_ZOMBIE classifications "
            "below are unreliable (a real crash loop can be masked as HEALTHY "
            "when the source that would reveal it is down)."
        ),
    )


__all__ = ["ModelRunnerFleetHealthVerdict"]
