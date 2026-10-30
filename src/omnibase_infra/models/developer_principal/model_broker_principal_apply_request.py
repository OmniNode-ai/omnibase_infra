# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for ``broker.principal.apply`` (OMN-20822).

It carries the plan and the *name* of the environment variable that holds the
password, never the password itself (mint plan D5).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.developer_principal.model_developer_principal_plan import (
    ModelDeveloperPrincipalPlan,
)


class ModelBrokerPrincipalApplyRequest(BaseModel):
    """Apply one planned principal: its ACLs, then its password."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan: ModelDeveloperPrincipalPlan = Field(
        ..., description="The plan compute node's output."
    )
    admin_url: str = Field(..., min_length=1, description="The Redpanda Admin API URL.")
    password_env: str = Field(
        ...,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="Name of the environment variable holding the new password.",
    )
