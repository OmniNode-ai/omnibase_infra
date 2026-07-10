# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-runner health assessment (OMN-13942) -- the COMPUTE node's classification output.

OMN-14228 Slice A adds the precondition data a remediation gate needs to fail
CLOSED on indeterminate health: per-runner source determinacy plus the typed
re-arm signals that today survive only as free text in ``detail``. This slice
does not add any executor or gate logic -- it only stops dropping data a
future gate would need.
"""

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
    is_determinate: bool = Field(
        default=True,
        description=(
            "False when the upstream GitHub or Docker source used to classify "
            "this runner failed (see ModelRunnerFleetSnapshot.github_source_ok/"
            "docker_source_ok). A downstream remediation gate MUST treat this "
            "state as unreliable -- never as a verified HEALTHY -- when False."
        ),
    )
    docker_restart_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Typed re-arm signal for CRASH_LOOPING (ModelRunnerFleetRunnerFact."
            "docker_restart_count at classification time). Carried as a typed "
            "field, not parsed out of `detail`, so a future idempotency key can "
            "key on the actual observed edge."
        ),
    )
    diag_heartbeat_age_seconds: float | None = Field(
        default=None,
        description=(
            "Typed re-arm signal for LISTENER_ZOMBIE (ModelRunnerFleetRunnerFact."
            "diag_heartbeat_age_seconds at classification time). None if the "
            "probe could not determine an age."
        ),
    )


__all__ = ["ModelRunnerHealthAssessment"]
