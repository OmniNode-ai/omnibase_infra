# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""What the routing decision saw, in the shape a later reader needs it.

Ticket: OMN-18412
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_visibility import (
    EnumCIRunnerRouteVisibility,
)


class ModelCIRunnerRouteEvidence(BaseModel):
    """What the decision saw, in the shape a later reader needs it."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    github_event: str = Field(description="Triggering event of the calling run.")
    repository: str = Field(description="Repository whose job was placed.")
    workflow_path: str = Field(description="Calling workflow path.")
    seam_json: str = Field(description="The ceiling this run carried.")
    visibility: EnumCIRunnerRouteVisibility = Field(
        description="Visibility the placement was judged against."
    )
    fleet_online: int | None = Field(default=None, ge=0)
    fleet_busy: int | None = Field(default=None, ge=0)
    idle: int | None = Field(
        default=None, description="online - busy, when both were read."
    )
    busy_fraction: float | None = Field(default=None, ge=0.0)
    lab_age_seconds: int | None = Field(default=None, ge=0)
    lab_error: str = Field(
        default="",
        description="Named cause when the lab reading was unusable, so a missing "
        "record and a malformed one are distinguishable.",
    )
    lab_host: str = Field(
        default="", description="The host whose reading refused the lab."
    )
    downgrade_refused_from: str = Field(
        default="",
        description="On a private-repository reversal, the capacity reason that was "
        "refused, so the record says what would have happened.",
    )
    violation: str = Field(
        default="", description="Text of a never-widen violation, when one occurred."
    )


__all__ = ["ModelCIRunnerRouteEvidence"]
