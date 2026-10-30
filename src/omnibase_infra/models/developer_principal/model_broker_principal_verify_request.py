# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for ``broker.principal.verify`` (OMN-20822)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.developer_principal.model_developer_principal_plan import (
    ModelDeveloperPrincipalPlan,
)


class ModelBrokerPrincipalVerifyRequest(BaseModel):
    """Verify one principal: its ACLs against the plan, then a login as it."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan: ModelDeveloperPrincipalPlan = Field(
        ..., description="The plan compute node's output."
    )
    admin_url: str = Field(..., min_length=1, description="The Redpanda Admin API URL.")
    password_env: str = Field(
        ...,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="Name of the environment variable holding the principal's password.",
    )
