# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The janitor's bounded reclaim/preserve decision for a single network."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.enum_network_disposition import (
    EnumNetworkDisposition,
)


class ModelNetworkDecision(BaseModel):
    """The janitor's decision for a single observed network.

    ``network_ref`` is the Docker network reference string (a 64-char hex ID),
    deliberately NOT a UUID — Docker assigns these, not ONEX.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    network_ref: str = Field(..., description="Docker network reference (hex ID)")
    name: str = Field(..., description="Docker network name")
    disposition: EnumNetworkDisposition = Field(
        ..., description="Bounded reclaim/preserve decision"
    )
    matched_rule: str = Field(
        default="",
        description="Name of the ownership rule that matched (empty if none)",
    )
    container_count: int = Field(
        default=0, ge=0, description="Containers attached at decision time"
    )
    age_seconds: int | None = Field(
        default=None, description="Network age in seconds; None if unknown"
    )
    reason: str = Field(default="", description="Human-readable rationale")


__all__ = ["ModelNetworkDecision"]
