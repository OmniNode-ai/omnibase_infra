# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-fleet snapshot -- the EFFECT's output model (OMN-13942).

Carries raw gh-api runner records, Docker inspect facts, queue/run-age facts,
and the two OMN-13932 probes (buildx availability, codeload-throttle
signature matches) gathered by ``node_runner_health_snapshot_effect``. This
model is facts-only; classification happens downstream in
``node_runner_fleet_health_compute``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_zombie_run_candidate import (
    ModelZombieRunCandidate,
)


class ModelRunnerFleetSnapshot(BaseModel):
    """Point-in-time, facts-only snapshot of the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    collected_at: datetime = Field(..., description="When the snapshot was collected.")
    host: str = Field(..., description="Runner host address.")
    expected_count: int = Field(
        ..., ge=0, description="Configured runner count (config/runner_fleet.yaml)."
    )
    runners: tuple[ModelRunnerFleetRunnerFact, ...] = Field(
        default_factory=tuple, description="Per-runner raw facts."
    )
    oldest_queued_job_age_seconds: float | None = Field(
        default=None,
        description=(
            "Age in seconds of the oldest queued self-hosted job across the "
            "watched repos. None if no job is queued or the probe failed."
        ),
    )
    zombie_run_candidates: tuple[ModelZombieRunCandidate, ...] = Field(
        default_factory=tuple,
        description="Queued/in-progress runs observed past the wedge-age threshold.",
    )
    buildx_available: bool | None = Field(
        default=None,
        description=(
            "Docker buildx availability on the runner host (OMN-13932). "
            "None means the probe could not determine availability."
        ),
    )
    codeload_throttle_signal_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Count of recent failed-run log signatures matching known "
            "codeload.github.com throttling patterns (OMN-13932)."
        ),
    )
    codeload_throttle_examples: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Short repo#run_id: matched-signature descriptors, bounded sample.",
    )
    github_source_ok: bool = Field(
        default=True, description="Whether the GitHub API source succeeded."
    )
    docker_source_ok: bool = Field(
        default=True, description="Whether the SSH/Docker inspection source succeeded."
    )
    source_errors: tuple[str, ...] = Field(
        default_factory=tuple, description="Error details for any failed sources."
    )


__all__ = ["ModelRunnerFleetSnapshot"]
