# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Input model for the onboarding orchestrator node."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelOnboardingInput(BaseModel):
    """Input for the onboarding orchestrator.

    When ``policy_name`` is set and the resolved policy has
    ``policy_type == "interactive"``, the handler dispatches to the
    interactive executor.  Otherwise the existing DAG-based
    ``resolve_policy`` + verification path is used.

    The ``input_adapter`` is injected via handler function parameter,
    NOT in this model (DI outside models — OMN-10784 GPT #1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    @model_validator(mode="after")
    def _enforce_env_output_path_when_writing(self) -> ModelOnboardingInput:
        """Reject dry_run=False with no env_output_path at construction time."""
        if not self.dry_run and (
            self.env_output_path is None or not self.env_output_path.strip()
        ):
            msg = "env_output_path is required when dry_run=False"
            raise ValueError(msg)
        return self


__all__ = ["ModelOnboardingInput"]
