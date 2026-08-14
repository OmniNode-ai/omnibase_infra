# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Phased state for one application database enforcement gate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelApplicationDatabaseEnforcementGateState(BaseModel):
    """Mandatory source proof and honest deployment activation state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_enforcement: Literal["mandatory"]
    deployment_enforcement: Literal["mandatory", "blocked"]
    source_proofs: tuple[str, ...] = Field(..., min_length=1)
    source_proof_paths: tuple[str, ...] = Field(..., min_length=1)
    seeded_red_controls: tuple[str, ...] = Field(..., min_length=1)
    deployment_blockers: tuple[str, ...] = ()

    @field_validator(
        "source_proofs",
        "source_proof_paths",
        "seeded_red_controls",
        "deployment_blockers",
    )
    @classmethod
    def validate_unique_nonempty_values(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Reject duplicates and whitespace-only evidence claims."""
        if len(set(values)) != len(values):
            raise ValueError("gate evidence values must be unique")
        if any(not value.strip() for value in values):
            raise ValueError("gate evidence values cannot be blank")
        return values

    @field_validator("source_proof_paths")
    @classmethod
    def validate_repository_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Keep proof paths repository-relative and traversal-free."""
        invalid = sorted(
            path for path in paths if path.startswith("/") or ".." in path.split("/")
        )
        if invalid:
            raise ValueError(
                f"source proof paths must be repository-relative: {invalid}"
            )
        return paths

    @model_validator(mode="after")
    def validate_deployment_state(self) -> ModelApplicationDatabaseEnforcementGateState:
        """A blocked deployment needs reasons; a mandatory one cannot retain them."""
        if self.deployment_enforcement == "blocked" and not self.deployment_blockers:
            raise ValueError("blocked deployment enforcement requires blockers")
        if self.deployment_enforcement == "mandatory" and self.deployment_blockers:
            raise ValueError("mandatory deployment enforcement cannot retain blockers")
        return self


__all__ = ["ModelApplicationDatabaseEnforcementGateState"]
