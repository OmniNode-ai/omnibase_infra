# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of ``broker.principal.apply`` (OMN-20822). It has no password field."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.developer_principal.model_broker_acl_binding import (
    ModelBrokerAclBinding,
)


class ModelBrokerPrincipalApplyResult(BaseModel):
    """What the apply did: create, rotate, or a drift refusal with the extras named."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    principal: str = Field(..., min_length=1, description="The principal.")
    action: Literal["create", "rotate", "refused"] = Field(..., description="Outcome.")
    acls_created: int = Field(default=0, ge=0, description="ACL bindings created now.")
    acls_applied: bool = Field(
        default=False, description="Every planned ACL read back."
    )
    password_set: bool = Field(default=False, description="The password was set.")
    extra_acls: tuple[ModelBrokerAclBinding, ...] = Field(
        default_factory=tuple, description="ACLs outside the plan (drift)."
    )
    refusal: str = Field(default="", description="Why the apply was refused.")
