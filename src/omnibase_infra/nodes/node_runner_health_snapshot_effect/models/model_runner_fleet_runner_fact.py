# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-runner raw facts gathered by the runner-fleet snapshot effect (OMN-13942)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerFleetRunnerFact(BaseModel):
    """Raw, pre-classification facts for a single self-hosted runner.

    This is deliberately a FACTS-ONLY model -- no health-state judgment is
    made here. Classification is the health COMPUTE node's job
    (``node_runner_fleet_health_compute``); this EFFECT only reports what it
    observed from the GitHub API and SSH/Docker inspection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ..., description="Runner container/registration name (canonical identity key)."
    )
    github_status: str = Field(
        ..., description="GitHub API status: 'online', 'offline', or 'not_registered'."
    )
    github_busy: bool = Field(
        ..., description="Whether GitHub reports a job in flight."
    )
    docker_status: str = Field(
        default="",
        description="Docker container state (running/restarting/not_found/...).",
    )
    docker_uptime: str = Field(default="", description="Docker ps status string.")
    docker_restart_count: int = Field(
        default=0, ge=0, description="Docker RestartCount (crash-loop signal)."
    )
    diag_heartbeat_age_seconds: float | None = Field(
        default=None,
        description=(
            "Age of the newest Runner.Listener _diag heartbeat file in seconds. "
            "None means the age could not be determined (probe failure or no "
            "_diag directory observed)."
        ),
    )
    stale_registration: bool = Field(
        default=False,
        description="True if a Docker container exists with no matching GitHub registration.",
    )
    error: str = Field(default="", description="Per-runner probe error detail, if any.")


__all__ = ["ModelRunnerFleetRunnerFact"]
