# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A request to acquire a lease on one resource of a kind (OMN-21424)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLeaseAcquireRequest(BaseModel):
    """``request_id`` makes the acquire idempotent: a repeat returns the same grant."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str = Field(min_length=1)
    resource: str = Field(min_length=1, pattern=r"^[^/]+$")
    holder: str = Field(min_length=1)
    request_id: UUID
    ttl_s: int | None = Field(default=None, ge=2)


__all__ = ["ModelLeaseAcquireRequest"]
