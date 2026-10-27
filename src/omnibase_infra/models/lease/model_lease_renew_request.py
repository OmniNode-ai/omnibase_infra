# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A request to renew one held lease (OMN-21424)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLeaseRenewRequest(BaseModel):
    """Keep the grant ``lease_id`` alive; the fence does not change."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    resource: str = Field(min_length=1, pattern=r"^[^/]+$")
    lease_id: UUID


__all__ = ["ModelLeaseRenewRequest"]
