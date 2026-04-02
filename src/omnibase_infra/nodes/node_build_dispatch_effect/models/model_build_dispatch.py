# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build dispatch effect models.

Related:
    - OMN-7318: node_build_dispatch_effect
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_buildability import EnumBuildability


class ModelBuildTarget(BaseModel):
    """A single ticket targeted for build dispatch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str = Field(..., description="Linear ticket identifier.")
    title: str = Field(..., description="Ticket title.")
    buildability: EnumBuildability = Field(..., description="Buildability classification.")


class ModelBuildDispatchInput(BaseModel):
    """Input to the build dispatch effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    targets: tuple[ModelBuildTarget, ...] = Field(
        ..., description="Tickets to dispatch for building."
    )
    dry_run: bool = Field(default=False, description="Skip actual dispatch.")


class ModelBuildDispatchOutcome(BaseModel):
    """Outcome for a single dispatched ticket."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str = Field(..., description="Linear ticket identifier.")
    dispatched: bool = Field(..., description="Whether dispatch succeeded.")
    error: str | None = Field(default=None, description="Error if dispatch failed.")


class ModelBuildDispatchResult(BaseModel):
    """Result from the build dispatch effect node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Build loop cycle correlation ID.")
    outcomes: tuple[ModelBuildDispatchOutcome, ...] = Field(
        ..., description="Per-ticket dispatch outcomes."
    )
    total_dispatched: int = Field(default=0, ge=0, description="Successfully dispatched count.")
    total_failed: int = Field(default=0, ge=0, description="Failed dispatch count.")


__all__: list[str] = [
    "ModelBuildDispatchInput",
    "ModelBuildDispatchOutcome",
    "ModelBuildDispatchResult",
    "ModelBuildTarget",
]
