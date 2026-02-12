# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Git repository state snapshot for RRH validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRRHRepoState(BaseModel):
    """Git repository state snapshot.

    Attributes:
        branch: Current checked-out branch name.
        head_sha: Full SHA of HEAD commit.
        is_dirty: Whether the working tree has uncommitted changes.
        repo_root: Absolute path to repository root.
        remote_url: Origin remote URL (empty string if no remote).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    branch: str = Field(..., description="Current branch name.")
    head_sha: str = Field(..., description="Full SHA of HEAD commit.")
    is_dirty: bool = Field(..., description="Working tree has uncommitted changes.")
    repo_root: str = Field(..., description="Absolute path to repo root.")
    remote_url: str = Field(default="", description="Origin remote URL.")


__all__: list[str] = ["ModelRRHRepoState"]
