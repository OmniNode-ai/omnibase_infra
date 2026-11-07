# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed request for the sequential setup workflow."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology


class ModelSetupOrchestratorRequest(BaseModel):
    """Deployment topology and tracing context for setup orchestration."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topology: ModelDeploymentTopology
    correlation_id: UUID
    compose_file_path: str


__all__: list[str] = ["ModelSetupOrchestratorRequest"]
