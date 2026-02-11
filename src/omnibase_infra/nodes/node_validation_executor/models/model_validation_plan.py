# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation plan model — input for the validation executor.

This model defines the plan of checks to execute. It is consumed by the
validation executor effect node and will also be produced by the
validation orchestrator (when that node is created).

Note:
    This is the canonical definition until ``node_validation_orchestrator``
    is created. At that point this model may move to the orchestrator's
    models package and be re-exported here for backwards compatibility.

Ticket: OMN-2147
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_validation_executor.models.model_planned_check import (
    ModelPlannedCheck,
)


class ModelValidationPlan(BaseModel):
    """Plan of validation checks to execute.

    Produced by the orchestrator and consumed by the executor effect node.

    Attributes:
        plan_id: Unique identifier for this plan.
        candidate_id: Reference to the pattern candidate being validated.
        checks: Tuple of planned checks to execute.
        executor_type: Executor type hint (e.g., "smoke", "full", "ci").
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan_id: UUID = Field(default_factory=uuid4, description="Unique plan identifier.")
    candidate_id: UUID = Field(..., description="Reference to the pattern candidate.")
    checks: tuple[ModelPlannedCheck, ...] = Field(
        default_factory=tuple, description="Planned checks to execute."
    )
    executor_type: str = Field(
        default="smoke", description="Executor type hint (smoke, full, ci)."
    )


__all__: list[str] = ["ModelValidationPlan"]
