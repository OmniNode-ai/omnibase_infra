# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner health snapshot model — a point-in-time view of all runners."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.model_runner_status import (
    ModelRunnerStatus,
)


class ModelRunnerHealthSnapshot(BaseModel):
    """Point-in-time health snapshot for all monitored runners.

    Separates ``expected_runners`` (configured count) from
    ``observed_runners`` (actually discovered) to prevent semantic
    overloading.  Tracks per-source failure via ``github_source_ok``,
    ``docker_source_ok``, and ``source_errors``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Trace correlation ID")
    collected_at: datetime = Field(..., description="When the snapshot was collected")
    runners: tuple[ModelRunnerStatus, ...] = Field(
        default_factory=tuple, description="Per-runner status"
    )
    expected_runners: int = Field(
        ..., description="Configured runner count (from env/config)", ge=0
    )
    observed_runners: int = Field(
        ..., description="Runners actually discovered from sources", ge=0
    )
    healthy_count: int = Field(..., description="Runners in HEALTHY state", ge=0)
    degraded_count: int = Field(..., description="Runners NOT in HEALTHY state", ge=0)
    github_source_ok: bool = Field(
        default=True, description="Whether GitHub API call succeeded"
    )
    docker_source_ok: bool = Field(
        default=True, description="Whether SSH Docker inspection succeeded"
    )
    source_errors: tuple[str, ...] = Field(
        default_factory=tuple, description="Error details for failed sources"
    )
    host: str = Field(..., description="CI host address")
    host_disk_percent: float = Field(
        default=0.0, description="Host disk usage percentage"
    )
