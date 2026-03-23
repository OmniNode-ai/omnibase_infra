# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner health snapshot model -- frozen point-in-time view of all runners."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.model_runner_status import (
    ModelRunnerStatus,
)


class ModelRunnerHealthSnapshot(BaseModel):
    """Frozen point-in-time health snapshot of all self-hosted runners.

    ``expected_runners`` (configured) and ``observed_runners`` (discovered)
    are modeled separately to prevent semantic overloading.  Source failure
    tracking (``github_source_ok``, ``docker_source_ok``, ``source_errors``)
    ensures partial-source degradation is surfaced explicitly.
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


__all__ = ["ModelRunnerHealthSnapshot"]
