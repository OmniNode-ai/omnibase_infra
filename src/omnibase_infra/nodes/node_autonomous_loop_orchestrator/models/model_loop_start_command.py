# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ModelLoopStartCommand — command to start the autonomous build loop.

Related:
    - OMN-7319: node_autonomous_loop_orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLoopStartCommand(BaseModel):
    """Command to start the autonomous build loop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Unique cycle ID.")
    max_cycles: int = Field(default=1, ge=1, description="Max cycles to run.")
    skip_closeout: bool = Field(
        default=False, description="Skip the CLOSING_OUT phase."
    )
    dry_run: bool = Field(default=False, description="No side effects if true.")
    requested_at: datetime = Field(..., description="When the command was issued.")


__all__: list[str] = ["ModelLoopStartCommand"]
