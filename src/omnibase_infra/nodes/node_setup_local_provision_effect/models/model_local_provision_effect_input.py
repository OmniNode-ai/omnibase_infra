# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Input model for the setup local provision effect node.

Ticket: OMN-3491
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology


class ModelLocalProvisionEffectInput(BaseModel):
    """Input envelope for the setup local provision effect node.

    Controls Docker Compose-based local service provisioning, teardown,
    or status checking based on the supplied topology.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topology: ModelDeploymentTopology = Field(
        ...,
        description="Deployment topology defining which local services to provision.",
    )
    compose_file_path: str = Field(
        ...,
        description="Path to the Docker Compose file used for provisioning.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for tracing.",
    )
    operation: Literal["provision_local", "teardown_local", "status_check"] = Field(
        ...,
        description=(
            "Operation to perform: "
            "provision_local (start services), "
            "teardown_local (stop and remove), "
            "status_check (query running state)."
        ),
    )
    max_wait_seconds: int = Field(
        default=120,
        ge=1,
        description="Maximum seconds to wait for services to become healthy.",
    )


__all__: list[str] = ["ModelLocalProvisionEffectInput"]
