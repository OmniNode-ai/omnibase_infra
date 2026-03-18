# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Onboarding verification spec within a graph step definition."""

from pydantic import BaseModel, Field


class ModelOnboardingStepVerification(BaseModel):
    """Verification spec within an onboarding graph step."""

    check_type: str = Field(description="Type of verification check")
    target: str = Field(description="Check-specific target")
    timeout_seconds: int = Field(default=10, description="Check timeout")
    description: str | None = Field(default=None, description="Human description")


__all__ = ["ModelOnboardingStepVerification"]
