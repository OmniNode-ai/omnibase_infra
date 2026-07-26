# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command model that carries a gathered snapshot to the health COMPUTE node (OMN-13942)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)


class ModelRunnerFleetHealthEvaluateCommand(BaseModel):
    """Command to classify a previously-gathered runner-fleet snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    snapshot: ModelRunnerFleetSnapshot = Field(
        ..., description="Facts-only snapshot to classify (no probing here)."
    )
