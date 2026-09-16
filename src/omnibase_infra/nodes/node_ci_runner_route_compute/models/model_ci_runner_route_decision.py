# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The emitted routing decision, complete enough to audit without the run log.

Ticket: OMN-18412
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_decision import (
    EnumCIRunnerRouteDecision,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_reason import (
    EnumCIRunnerRouteReason,
)
from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_evidence import (
    ModelCIRunnerRouteEvidence,
)


class ModelCIRunnerRouteDecision(BaseModel):
    """The emitted decision, complete enough to audit without the run log."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    runs_on: tuple[str, ...] = Field(
        description="The label set the calling workflow consumes. EMPTY on a "
        "refusal: a refusal names no placement, which is not the same as hosted."
    )
    decision: EnumCIRunnerRouteDecision = Field(description="Placement class chosen.")
    reason: EnumCIRunnerRouteReason = Field(description="Why.")
    reason_detail: str = Field(
        default="",
        description="Qualifier for a reason that has classes, such as the probe "
        "failure class or the cause a refusal is refusing.",
    )
    policy_version: int = Field(
        gt=0, description="Version of the policy that produced this decision."
    )
    decided_at: str = Field(description="ISO-8601 UTC timestamp of the decision.")
    evidence: ModelCIRunnerRouteEvidence = Field(description="What it saw.")

    @property
    def reason_wire(self) -> str:
        """The reason as the workflow, the audit and the monitor read it.

        Kept as ONE renderer rather than spelled at each call site: the wire
        form is a consumed interface -- the saturation monitor counts these
        strings and the routing audit recognises a routed job by them -- and a
        second spelling of it is a silent break in whichever consumer reads the
        other one.
        """
        if self.reason_detail:
            return f"{self.reason.value}:{self.reason_detail}"
        return self.reason.value


__all__ = ["ModelCIRunnerRouteDecision"]
