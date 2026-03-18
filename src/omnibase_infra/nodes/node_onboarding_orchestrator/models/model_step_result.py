# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Per-step result model for the onboarding orchestrator."""

from pydantic import BaseModel, Field


class ModelStepResult(BaseModel):
    """Result of executing a single onboarding step."""

    step_key: str = Field(description="Step key that was executed")
    passed: bool = Field(description="Whether the step passed")
    message: str = Field(description="Result message")
    elapsed_ms: int = Field(default=0, description="Execution time in ms")


__all__ = ["ModelStepResult"]
