# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Top-level onboarding graph model parsed from YAML."""

from pydantic import BaseModel, Field

from omnibase_infra.onboarding.model_onboarding_step import ModelOnboardingStep


class ModelOnboardingGraph(BaseModel):
    """Top-level onboarding graph model parsed from YAML."""

    title: str = Field(description="Human-readable graph title")
    description: str | None = Field(default=None, description="Graph description")
    steps: list[ModelOnboardingStep] = Field(description="Ordered list of graph steps")


__all__ = ["ModelOnboardingGraph"]
