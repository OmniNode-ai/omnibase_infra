# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Input model for the setup Infisical effect node.

Ticket: OMN-3491
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology


class ModelInfisicalSetupEffectInput(BaseModel):
    """Input envelope for the setup Infisical effect node.

    Manages Infisical secret store provisioning and seeding as part of
    the setup orchestration workflow.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topology: ModelDeploymentTopology = Field(
        ...,
        description="Deployment topology; infisical service config is read from it.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for tracing.",
    )
    infisical_addr: str | None = Field(
        default=None,
        description=(
            "Override Infisical address. If None, the address is resolved "
            "from the topology local config."
        ),
    )
    skip_if_disabled: bool = Field(
        default=True,
        description=(
            "If True and infisical service mode is DISABLED, "
            "skip provisioning and emit setup.infisical.skipped event."
        ),
    )


__all__: list[str] = ["ModelInfisicalSetupEffectInput"]
