# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-local deploy agent image maintenance request (OMN-19998)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDeployAgentImagePruneRequest(BaseModel):
    """Planning is the default; installation and removal are explicit operations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal[
        "deploy_agent.images.plan",
        "deploy_agent.images.prune",
        "deploy_agent.images.install",
    ] = "deploy_agent.images.plan"
    policy_path: str
    repository: str = "."
