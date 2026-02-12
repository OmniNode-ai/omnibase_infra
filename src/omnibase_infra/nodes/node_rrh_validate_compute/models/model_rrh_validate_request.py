# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Input model for the RRH validate compute node."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.rrh.model_rrh_environment_data import (
    ModelRRHEnvironmentData,
)
from omnibase_infra.nodes.node_rrh_validate_compute.models.model_rrh_contract_governance import (
    ModelRRHContractGovernance,
)


class ModelRRHValidateRequest(BaseModel):
    """Request to validate environment data against RRH rules.

    Attributes:
        environment_data: Collected environment snapshot.
        profile_name: Profile to use (default, ticket-pipeline, ci-repair, seam-ticket).
        governance: Contract governance fields for rule tightening.
        repo_name: Repository name for result metadata.
        correlation_id: Distributed tracing correlation ID.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    environment_data: ModelRRHEnvironmentData = Field(
        ..., description="Collected environment snapshot."
    )
    profile_name: str = Field(default="default", description="Validation profile name.")
    governance: ModelRRHContractGovernance = Field(
        default_factory=ModelRRHContractGovernance,
        description="Contract governance fields.",
    )
    repo_name: str = Field(default="", description="Repository name.")
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for tracing."
    )


__all__: list[str] = ["ModelRRHValidateRequest"]
