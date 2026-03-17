# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Output model for the onboarding orchestrator node."""

from pydantic import BaseModel, Field

from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_step_result import (
    ModelStepResult,
)


class ModelOnboardingOutput(BaseModel):
    """Output from the onboarding orchestrator."""

    success: bool = Field(description="Whether all steps passed")
    total_steps: int = Field(description="Total number of steps")
    completed_steps: int = Field(description="Number of completed steps")
    step_results: list[ModelStepResult] = Field(description="Per-step results")
    rendered_output: str = Field(description="Rendered output string")


__all__ = ["ModelOnboardingOutput"]
