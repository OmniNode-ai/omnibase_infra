# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reducer's intent to upsert one lease projection row (OMN-21424)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.lease.model_lease_projection_row import (
    ModelLeaseProjectionRow,
)


class ModelPayloadLeaseProjectionUpsert(BaseModel):
    """Target ``postgres://lease_projection/{kind}/{resource}``."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["lease.projection.upsert"] = "lease.projection.upsert"
    row: ModelLeaseProjectionRow

    @property
    def target(self) -> str:
        return f"postgres://lease_projection/{self.row.kind}/{self.row.resource}"


__all__ = ["ModelPayloadLeaseProjectionUpsert"]
