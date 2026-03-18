# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Onboarding step model for graph definitions."""

from pydantic import BaseModel, Field

from omnibase_infra.onboarding.model_onboarding_step_verification import (
    ModelOnboardingStepVerification,
)


class ModelOnboardingStep(BaseModel):
    """A single step in the onboarding graph."""

    step_key: str = Field(description="Unique step key within the graph")
    name: str = Field(description="Human-readable step name")
    step_type: str = Field(description="Step type (verification, action)")
    description: str | None = Field(default=None, description="Step description")
    depends_on: list[str] = Field(
        default_factory=list, description="Dependency step keys"
    )
    required_capabilities: list[str] = Field(
        default_factory=list, description="Required capabilities"
    )
    produces_capabilities: list[str] = Field(
        default_factory=list, description="Capabilities produced"
    )
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    estimated_duration_seconds: int | None = Field(
        default=None, description="Estimated duration"
    )
    verification: ModelOnboardingStepVerification | None = Field(
        default=None, description="Proof condition"
    )


__all__ = ["ModelOnboardingStep"]
