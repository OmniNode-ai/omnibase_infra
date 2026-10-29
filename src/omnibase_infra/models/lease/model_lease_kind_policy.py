# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How many holders one resource of a lease kind admits, and its TTL (OMN-21424)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLeaseKindPolicy(BaseModel):
    """One lease kind: ``capacity`` concurrent grants per resource for ``ttl_s``."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    capacity: int = Field(default=1, ge=1, le=64)
    ttl_s: int = Field(default=30, ge=2)


__all__ = ["ModelLeaseKindPolicy"]
