# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One row of the lease projection, keyed on ``(kind, resource)`` (OMN-21424)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLeaseProjectionRow(BaseModel):
    """The latest grant of a resource; a row is never replaced by a lower fence."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str
    resource: str
    lease_id: UUID
    fence: int = Field(ge=1)
    holder: str
    state: Literal["held", "released", "expired"]


__all__ = ["ModelLeaseProjectionRow"]
