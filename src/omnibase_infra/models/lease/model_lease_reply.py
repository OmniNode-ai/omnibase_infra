# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lease orchestrator's answer to an acquire, renew or release (OMN-21424)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

LeaseOutcome = Literal[
    "lease-granted",
    "lease-refused",
    "lease-renewed",
    "lease-released",
    "lease-expired",
]


class ModelLeaseReply(BaseModel):
    """One outcome; ``fence`` is the etcd revision of the grant, 0 when none."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    outcome: LeaseOutcome
    kind: str
    resource: str
    holder: str = ""
    lease_id: UUID | None = None
    fence: int = Field(default=0, ge=0)
    request_id: UUID | None = None
    ttl_s: int = Field(default=0, ge=0)
    reason: str = ""


__all__ = ["LeaseOutcome", "ModelLeaseReply"]
