# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Everything the routing decision reads.

Ticket: OMN-18412
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_force import (
    EnumCIRunnerRouteForce,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_visibility import (
    EnumCIRunnerRouteVisibility,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_fleet_observation import (
    ModelCIRunnerFleetObservation,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_lab_observation import (
    ModelCIRunnerLabObservation,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_policy import (
    ModelCIRunnerRoutePolicy,
)


class ModelCIRunnerRouteRequest(BaseModel):
    """Everything the decision reads.

    ``seam_json`` is the trusted-CI runner variable's value and is a CEILING:
    routing may only ever DOWNGRADE from what it already allows, never widen.
    It is read and never written (CLAUDE.md rule 14, single-owner).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        default_factory=uuid4, description="Trace correlation ID."
    )
    github_event: str = Field(description="The triggering event of the calling run.")
    head_repo: str = Field(
        default="",
        description="Head repository of a pull request, for the fork test. Empty on "
        "every other event.",
    )
    repository: str = Field(
        description="The repository whose job is being placed. In a called reusable "
        "workflow this is the CALLER's repository, which is the one whose visibility "
        "and whose seam value govern the placement."
    )
    workflow_path: str = Field(
        description="Path of the CALLING workflow, checked against the hosted list."
    )
    seam_json: str = Field(
        default="",
        description="Raw value of the trusted-CI runner variable: the ceiling.",
    )
    public_json: str = Field(
        default="",
        description="Raw value of the public/fork runner variable.",
    )
    visibility: EnumCIRunnerRouteVisibility = Field(
        description="Visibility of `repository`, read live at decision time."
    )
    fleet: ModelCIRunnerFleetObservation = Field(
        description="Capacity reading of the runner group."
    )
    fleet_expected_count: int = Field(
        gt=0,
        description="How many runners the fleet INVENTORY declares should exist, "
        "read from config/runner_fleet.yaml rather than restated as a literal. The "
        "degraded floor is a fraction of this, so resizing the fleet moves the floor "
        "with it.",
    )
    lab: ModelCIRunnerLabObservation = Field(description="Lab-load reading.")
    hosted_workflows: tuple[str, ...] = Field(
        default=(),
        description="Workflow paths whose routed jobs stay hosted whatever the "
        "capacity, for reasons that outrank capacity: fate isolation from the fleet "
        "being watched, clean egress for a registry push, fork-only verification.",
    )
    force: EnumCIRunnerRouteForce = Field(
        default=EnumCIRunnerRouteForce.AUTO,
        description="Operator override, subject to every safety rule.",
    )
    policy: ModelCIRunnerRoutePolicy = Field(
        description="The declared thresholds, from the node's contract."
    )


__all__ = ["ModelCIRunnerRouteRequest"]
