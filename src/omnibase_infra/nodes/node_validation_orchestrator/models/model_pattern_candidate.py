# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pattern candidate model -- input to the validation orchestrator."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPatternCandidate(BaseModel):
    """A pattern candidate submitted for validation.

    Represents a code change or pattern that needs to go through
    the validation pipeline (typecheck, lint, tests, risk assessment).

    Attributes:
        candidate_id: Unique identifier for this candidate.
        pattern_id: Identifier of the pattern being validated.
        source_path: Root path of the code change.
        diff_summary: Brief description of what changed.
        changed_files: List of file paths that were modified.
        risk_tags: Tags indicating risk areas (e.g., "security", "api-change").
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    candidate_id: UUID = Field(..., description="Unique identifier for this candidate.")
    pattern_id: UUID = Field(
        ..., description="Identifier of the pattern being validated."
    )
    source_path: str = Field(..., description="Root path of the code change.")
    diff_summary: str = Field(
        default="", description="Brief description of what changed."
    )
    changed_files: tuple[str, ...] = Field(
        default_factory=tuple, description="File paths that were modified."
    )
    risk_tags: tuple[str, ...] = Field(
        default_factory=tuple, description="Risk area tags."
    )


__all__: list[str] = ["ModelPatternCandidate"]
