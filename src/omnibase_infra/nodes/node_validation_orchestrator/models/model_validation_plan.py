# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation plan model — canonical definition shared by orchestrator and executor.

Produced by the orchestrator's build_plan step and consumed by the executor
effect node.

Ticket: OMN-2147
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_validation_orchestrator.models.model_planned_check import (
    ModelPlannedCheck,
)


class ModelValidationPlan(BaseModel):
    """Validation plan produced by the orchestrator and consumed by the executor.

    Contains the ordered list of checks to execute for a given candidate.

    Attributes:
        plan_id: Unique identifier for this plan.
        candidate_id: Reference to the pattern candidate.
        checks: Ordered tuple of planned checks.
        score_threshold: Minimum score for PASS verdict (0.0-1.0).
        executor_type: Executor type hint (e.g., "smoke", "full", "ci").
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan_id: UUID = Field(default_factory=uuid4, description="Unique plan identifier.")
    candidate_id: UUID = Field(..., description="Reference to the pattern candidate.")
    checks: tuple[ModelPlannedCheck, ...] = Field(
        default_factory=tuple, description="Ordered list of checks to execute."
    )
    score_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum score for PASS verdict."
    )
    executor_type: str = Field(
        default="smoke", description="Executor type hint (smoke, full, ci)."
    )


__all__: list[str] = ["ModelValidationPlan"]
