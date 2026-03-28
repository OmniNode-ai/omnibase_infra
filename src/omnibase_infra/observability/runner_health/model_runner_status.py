# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner status model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)


class ModelRunnerStatus(BaseModel):
    """Status of a single self-hosted runner.

    ``name`` is the canonical identity key: it must match between
    GitHub registration names and Docker container names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ..., description="Runner container/registration name (canonical identity key)"
    )
    github_status: str = Field(..., description="GitHub API status: online/offline")
    github_busy: bool = Field(..., description="Whether runner is executing a job")
    docker_status: str = Field(..., description="Docker container health status")
    docker_uptime: str = Field(default="", description="Docker ps status string")
    state: EnumRunnerHealthState = Field(..., description="Computed health state")
    error: str = Field(default="", description="Error detail if degraded")
