# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A request to release one held lease (OMN-21424)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLeaseReleaseRequest(BaseModel):
    """Give the grant ``lease_id`` back; a release of a lost grant is a no-op.

    ``outcome`` is the holder's own result (a ``lane-work`` lane's TERMINAL
    outcome, OMN-21428); the release reply carries it in ``reason``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str = Field(min_length=1)
    resource: str = Field(min_length=1, pattern=r"^[^/]+$")
    lease_id: UUID
    outcome: str = Field(default="", pattern=r"^[a-z-]*$")


__all__ = ["ModelLeaseReleaseRequest"]
