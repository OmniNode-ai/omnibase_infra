# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command model that triggers a runner-fleet snapshot gather (OMN-13942)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerFleetSnapshotGatherCommand(BaseModel):
    """Command to gather a point-in-time runner-fleet snapshot.

    Thin by design: the fleet identity (host, org, expected count, watch
    repos) is authoritative in ``config/runner_fleet.yaml``, not carried on
    the command. Only the workflow correlation ID is threaded through.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
