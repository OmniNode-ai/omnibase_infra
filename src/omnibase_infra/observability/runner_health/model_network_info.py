# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker network info model — one observed Docker network on a runner host."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelNetworkInfo(BaseModel):
    """A single Docker network observed on a runner host.

    ``container_count`` distinguishes an idle network (safe to reclaim once
    stale) from one with attached containers (an active lane that must never
    be pruned). ``created_at`` drives the age filter; ``None`` means the
    creation timestamp could not be parsed and the network is treated as
    age-unknown (preserve).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    network_ref: str = Field(
        ..., description="Docker network reference (short or full hex ID, not a UUID)"
    )
    name: str = Field(..., description="Docker network name")
    created_at: datetime | None = Field(
        default=None,
        description="Network creation time (UTC); None if unparseable (age-unknown)",
    )
    container_count: int = Field(
        default=0, ge=0, description="Number of containers attached to this network"
    )
    is_builtin: bool = Field(
        default=False,
        description="Whether this is a Docker builtin network (bridge/host/none)",
    )


__all__ = ["ModelNetworkInfo"]
