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
    overlay_output_path: str | None = Field(
        default=None,
        description="Path for overlay YAML write; derived from env_output_path when None",
    )
    legacy_env_output: bool = Field(
        default=True,
        description="When True, also write legacy .env file alongside overlay output",
    )

    @model_validator(mode="after")
    def _enforce_env_output_path_when_writing(self) -> ModelOnboardingInput:
        """Reject write mode unless at least one requested output path is valid."""
        if self.dry_run:
            return self

        has_env_path = self.env_output_path is not None and bool(
            self.env_output_path.strip()
        )
        has_overlay_path = self.overlay_output_path is not None and bool(
            self.overlay_output_path.strip()
        )
        if self.env_output_path is not None and not has_env_path:
            msg = "env_output_path cannot be blank"
            raise ValueError(msg)
        if self.overlay_output_path is not None and not has_overlay_path:
            msg = "overlay_output_path cannot be blank"
            raise ValueError(msg)
        if self.legacy_env_output and not has_env_path:
            msg = "env_output_path is required when legacy_env_output=True"
            raise ValueError(msg)
        if not has_env_path and not has_overlay_path:
            msg = (
                "overlay_output_path is required when dry_run=False and "
                "legacy_env_output=False"
            )
            raise ValueError(msg)
        return self


__all__ = ["ModelOnboardingInput"]
