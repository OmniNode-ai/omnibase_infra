# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline model — execution request.

Published by the orchestrator on ``validation-execution-requested`` and
consumed by the executor effect node. It is a distinct class from
``ModelValidationPlan`` because the contract's ``published_events`` map routes a
returned model to its topic by class name: the plan itself is already the
``validation-plan-created`` event.

Ticket: OMN-21056
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_validation_plan import (
    ModelValidationPlan,
)


class ModelValidationExecutionRequest(BaseModel):
    """Request to the executor to run every check in a validation plan.

    Attributes:
        plan: The validation plan whose enabled checks the executor runs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan: ModelValidationPlan = Field(
        ..., description="The validation plan to execute."
    )


__all__: list[str] = ["ModelValidationExecutionRequest"]
