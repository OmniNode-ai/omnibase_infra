# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-resolved configuration, never accepted from a command or serialized."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_deploy_agent_branch_trial_policy import (
    ModelDeployAgentBranchTrialPolicy,
)


class ModelDeployAgentBranchTrialHost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    environment: dict[str, str] = Field(exclude=True, repr=False)
    allowed_lanes: str
    tracking_ref: str
    repo_root: Path
    state_dir: Path
    python: Path
    health_port: int = Field(ge=1, le=65535)
    home: Path
    registry_root: Path | None
    policy: ModelDeployAgentBranchTrialPolicy
