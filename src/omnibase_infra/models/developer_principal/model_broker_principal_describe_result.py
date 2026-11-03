# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of ``broker.principal.describe`` (OMN-20823)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.developer_principal.model_broker_acl_binding import (
    ModelBrokerAclBinding,
)


class ModelBrokerPrincipalDescribeResult(BaseModel):
    """Every ACL the broker holds for the principal, sorted by label.

    Every binding is kept as the broker reports it, a DENY or a host-scoped
    ALLOW included, so an audit can name each binding outside the plan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    principal: str = Field(..., min_length=1, description="The principal.")
    acls: tuple[ModelBrokerAclBinding, ...] = Field(
        default_factory=tuple, description="The principal's ACLs, sorted by label."
    )


__all__: list[str] = ["ModelBrokerPrincipalDescribeResult"]
