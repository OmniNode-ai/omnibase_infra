# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for interactive onboarding execution results.

OMN-10782 / Task 5 of the interactive-onboarding-executor plan.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from omnibase_infra.onboarding.model_step_result import ModelStepResult


class ModelInteractiveResult(BaseModel):
    """Complete result of an interactive onboarding execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    env_dict: dict[str, str] = Field(
        description="Environment variables produced by the terminal step"
    )
    credentials_dict: dict[str, SecretStr] = Field(
        default_factory=dict,
        description=(
            "Secret material produced by the terminal step, as "
            "{secret_ref: secret}. Held as SecretStr because this model is "
            "returned as onboarding provenance and is therefore repr'd, "
            "logged, and model-dumped into receipts (OMN-16038); only the "
            "credentials writer unwraps it."
        ),
    )
    step_results: list[ModelStepResult] = Field(
        description="Ordered list of step results from the execution"
    )
    policy_name: str  # ONEX_EXCLUDE: pattern_validator - policy_name is the policy's own identifier, not an entity reference
    completed: bool = Field(description="Whether execution reached a terminal step")
    terminal_step: str = Field(
        description="ID of the terminal step where execution ended"
    )


__all__ = ["ModelInteractiveResult", "ModelStepResult"]
