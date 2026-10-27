# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The plan for one developer machine's broker principal (OMN-20821)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.developer_principal.model_developer_principal_grant import (
    ModelDeveloperPrincipalGrant,
)


class ModelDeveloperPrincipalPlan(BaseModel):
    """Create or rotate a named principal with an exact grant list, or a refusal.

    A refusal names its kind: ``declaration`` (the grants declaration breaks a
    rule the lab issuer enforces), ``identity`` (the login or device yields no
    name, or the login is outside the allowed domains) or ``device_cap`` (the
    prefix already holds as many principals as one login may).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    decision: Literal["create", "rotate", "refused"] = Field(
        ..., description="What to do."
    )
    principal: str = Field(default="", description="The planned principal name.")
    lane: str = Field(default="", description="The declaration's lane.")
    grants: tuple[ModelDeveloperPrincipalGrant, ...] = Field(
        default=(),
        description="The exact grants, sorted by resource, name and pattern.",
    )
    declaration_digest: str = Field(
        default="", description="sha256 of the declaration and its derived documents."
    )
    refusal_kind: Literal["", "declaration", "identity", "device_cap"] = Field(
        default="", description="Why the plan was refused; empty when it was not."
    )
    refusal: str = Field(default="", description="The refusal, naming the rule.")


__all__: list[str] = ["ModelDeveloperPrincipalPlan"]
