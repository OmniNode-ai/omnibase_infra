# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Autonomous loop orchestrator models.

Related:
    - OMN-7319: node_autonomous_loop_orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase


class ModelLoopStartCommand(BaseModel):
    """Command to start the autonomous build loop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Unique cycle ID.")
    max_cycles: int = Field(default=1, ge=1, description="Max cycles to run.")
    skip_closeout: bool = Field(default=False, description="Skip the CLOSING_OUT phase.")
    dry_run: bool = Field(default=False, description="No side effects if true.")
    requested_at: datetime = Field(..., description="When the command was issued.")


class ModelLoopCycleSummary(BaseModel):
    """Summary of a completed build loop cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Cycle correlation ID.")
    cycle_number: int = Field(..., ge=1, description="Cycle number.")
    final_phase: EnumBuildLoopPhase = Field(..., description="Terminal phase reached.")
    started_at: datetime = Field(..., description="Cycle start time.")
    completed_at: datetime = Field(..., description="Cycle completion time.")
    tickets_filled: int = Field(default=0, ge=0)
    tickets_classified: int = Field(default=0, ge=0)
    tickets_dispatched: int = Field(default=0, ge=0)
    error_message: str | None = Field(default=None)


class ModelLoopOrchestratorResult(BaseModel):
    """Final result from the autonomous loop orchestrator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Root correlation ID.")
    cycles_completed: int = Field(default=0, ge=0, description="Cycles completed.")
    cycles_failed: int = Field(default=0, ge=0, description="Cycles that failed.")
    cycle_summaries: tuple[ModelLoopCycleSummary, ...] = Field(
        default_factory=tuple, description="Per-cycle summaries."
    )
    total_tickets_dispatched: int = Field(
        default=0, ge=0, description="Total tickets dispatched across all cycles."
    )


__all__: list[str] = [
    "ModelLoopCycleSummary",
    "ModelLoopOrchestratorResult",
    "ModelLoopStartCommand",
]
