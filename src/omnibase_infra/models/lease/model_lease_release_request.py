# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A request to release one held lease (OMN-21424)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLeaseReleaseRequest(BaseModel):
    """Give the grant ``lease_id`` back; a release of a lost grant is a no-op."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str = Field(min_length=1)
    resource: str = Field(min_length=1, pattern=r"^[^/]+$")
    lease_id: UUID


__all__ = ["ModelLeaseReleaseRequest"]
