# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for a single interactive onboarding step result.

OMN-10782 / Task 5 of the interactive-onboarding-executor plan.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelStepResult(BaseModel):
    """Result of executing a single interactive onboarding step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_key: str = Field(description="Step identifier (= ModelInteractiveStep.id)")
    step_title: str = Field(description="User-facing prompt text of the step")
    response: str | list[str] = Field(
        description="User response — str for choice/text, list for multi_choice"
    )


__all__ = ["ModelStepResult"]
