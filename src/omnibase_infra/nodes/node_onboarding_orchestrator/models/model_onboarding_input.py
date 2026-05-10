# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for the onboarding orchestrator node."""

from pydantic import BaseModel, Field


class ModelOnboardingInput(BaseModel):
    """Input for the onboarding orchestrator.

    When ``policy_name`` is set and the resolved policy has
    ``policy_type == "interactive"``, the handler dispatches to the
    interactive executor.  Otherwise the existing DAG-based
    ``resolve_policy`` + verification path is used.

    The ``input_adapter`` is injected via handler function parameter,
    NOT in this model (DI outside models — OMN-10784 GPT #1).
    """

    target_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities to achieve (used by DAG path)",
    )
    skip_steps: list[str] = Field(
        default_factory=list,
        description="Step keys to skip (DAG path only)",
    )
    continue_on_failure: bool = Field(
        default=False,
        description="Whether to continue after a step fails (DAG path only)",
    )
    policy_name: str | None = Field(
        default=None,
        description="Policy name for interactive dispatch; None = existing DAG behavior",
    )
    dry_run: bool = Field(
        default=True,
        description="If true, return env output without writing to disk",
    )
    env_output_path: str | None = Field(
        default=None,
        description="Explicit path for env write; required when dry_run=False",
    )


__all__ = ["ModelOnboardingInput"]
