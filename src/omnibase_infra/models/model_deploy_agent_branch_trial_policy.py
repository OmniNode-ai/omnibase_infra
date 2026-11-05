# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-owned host supervision policy."""

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployAgentBranchTrialPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    health_host: str
    health_timeout_seconds: float = Field(gt=0)
    git_timeout_seconds: float = Field(gt=0)
    stop_timeout_seconds: float = Field(gt=0)
    heartbeat_timeout_seconds: float = Field(gt=0)
