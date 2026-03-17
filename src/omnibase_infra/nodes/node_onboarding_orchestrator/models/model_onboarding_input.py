# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for the onboarding orchestrator node."""

from pydantic import BaseModel, Field


class ModelOnboardingInput(BaseModel):
    """Input for the onboarding orchestrator."""

    target_capabilities: list[str] = Field(description="Capabilities to achieve")
    skip_steps: list[str] = Field(
        default_factory=list,
        description="Step keys to skip",
    )
    continue_on_failure: bool = Field(
        default=False,
        description="Whether to continue after a step fails",
    )


__all__ = ["ModelOnboardingInput"]
