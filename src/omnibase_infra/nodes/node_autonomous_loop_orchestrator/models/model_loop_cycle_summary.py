# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelLoopCycleSummary — summary of a completed build loop cycle.

Related:
    - OMN-7319: node_autonomous_loop_orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase


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


__all__: list[str] = ["ModelLoopCycleSummary"]
