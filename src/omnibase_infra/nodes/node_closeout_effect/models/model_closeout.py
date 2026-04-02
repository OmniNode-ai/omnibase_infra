# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Closeout effect models.

Related:
    - OMN-7316: node_closeout_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelCloseoutInput(BaseModel):
    """Input to the closeout effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    dry_run: bool = Field(default=False, description="Skip actual side effects.")


class ModelCloseoutResult(BaseModel):
    """Result from the closeout effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    merge_sweep_completed: bool = Field(
        default=False, description="Whether merge-sweep ran successfully."
    )
    prs_merged: int = Field(default=0, ge=0, description="PRs auto-merged.")
    quality_gates_passed: bool = Field(
        default=False, description="Whether quality gates passed."
    )
    release_ready: bool = Field(
        default=False, description="Whether release readiness check passed."
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple, description="Non-fatal warnings."
    )


__all__: list[str] = ["ModelCloseoutInput", "ModelCloseoutResult"]
