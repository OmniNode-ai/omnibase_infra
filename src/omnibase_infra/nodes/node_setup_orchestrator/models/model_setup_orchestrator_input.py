# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Input model for the setup orchestrator node.

Ticket: OMN-3491
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology


class ModelSetupOrchestratorInput(BaseModel):
    """Input envelope for the setup orchestrator node.

    The orchestrator drives the full setup workflow: preflight → provision
    → infisical → validate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topology: ModelDeploymentTopology = Field(
        ...,
        description="Deployment topology driving the full setup workflow.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for tracing across all child effect nodes.",
    )
    compose_file_path: str = Field(
        ...,
        description="Path to the Docker Compose file used by the provision effect node.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, emit events without performing actual provisioning.",
    )


__all__: list[str] = ["ModelSetupOrchestratorInput"]
