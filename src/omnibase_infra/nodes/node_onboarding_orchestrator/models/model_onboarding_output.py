# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Output model for the onboarding orchestrator node."""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_infra.nodes.node_onboarding_orchestrator.models.enum_onboarding_status import (
    EnumOnboardingStatus,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_step_result import (
    ModelStepResult,
)
from omnibase_infra.onboarding.model_interactive_result import ModelInteractiveResult


class ModelOnboardingOutput(BaseModel):
    """Output from the onboarding orchestrator.

    For DAG-based runs, only the base fields are populated.
    For interactive runs, ``provenance`` carries the full
    ``ModelInteractiveResult`` and the extra metadata fields are set.
    """

    success: bool = Field(description="Whether all steps passed")
    total_steps: int = Field(description="Total number of steps")
    completed_steps: int = Field(description="Number of completed steps")
    step_results: list[ModelStepResult] = Field(description="Per-step results")
    rendered_output: str = Field(description="Rendered output string")
    status: EnumOnboardingStatus = Field(
        default=EnumOnboardingStatus.FAILED,
        description="Overall onboarding result",
    )
    verified_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities that passed verification",
    )
    unmet_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities that failed verification",
    )

    # Interactive provenance (OMN-10784 GPT #11)
    provenance: ModelInteractiveResult | None = Field(
        default=None,
        description="Full interactive executor result — None for DAG-based runs",
    )
    policy_name: str | None = Field(
        default=None,
        description="Name of the policy that was executed",
    )
    policy_type: str | None = Field(
        default=None,
        description="Type of the policy (interactive or dag)",
    )
    visited_steps: list[str] = Field(
        default_factory=list,
        description="Step IDs visited during interactive execution",
    )
    terminal_step: str | None = Field(
        default=None,
        description="ID of the terminal step where interactive execution ended",
    )
    dry_run: bool = Field(
        default=True,
        description="Whether the run was dry (no file writes)",
    )
    env_output_path_written: str | None = Field(
        default=None,
        description="Path written to if dry_run=False; None otherwise",
    )
    overlay_output_path_written: str | None = Field(
        default=None,
        description="Path where overlay YAML was written, if generated",
    )


__all__ = ["ModelOnboardingOutput"]
