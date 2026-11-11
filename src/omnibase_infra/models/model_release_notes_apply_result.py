# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Result model for placing a rendered release block (OMN-17427)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelReleaseNotesApplyResult(BaseModel):
    """Repo-relative paths the cut job stages, or the reason the node refused."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(..., description="False when the node refused to write.")
    target: str = Field(
        default="",
        description="Where the policy sends the notes: CHANGELOG.md or README.md.",
    )
    staged: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Repo-relative paths to `git add`, in staging order.",
    )
    error_message: str | None = Field(
        default=None,
        description="Why the node refused; set only when success is false.",
    )
