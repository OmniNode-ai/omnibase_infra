# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lab lease policy and the inventory mechanism it replaces (OMN-21427)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.lease.model_lease_kind_policy import ModelLeaseKindPolicy


class ModelLeaseKindDeclaration(BaseModel):
    """One inventory row's declared lease policy and replaced mechanism."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    policy: ModelLeaseKindPolicy
    inventory_row: str
    replaces: str = Field(min_length=1)


__all__ = ["ModelLeaseKindDeclaration"]
