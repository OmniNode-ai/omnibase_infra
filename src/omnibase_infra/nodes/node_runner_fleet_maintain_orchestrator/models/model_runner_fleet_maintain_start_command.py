# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Start command for the runner-fleet-maintain workflow (OMN-13942)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerFleetMaintainStartCommand(BaseModel):
    """Command that starts one runner-fleet-maintain tick.

    Triggered by the reused OMN-13915 ``runner-fleet-canary`` 15-min
    GitHub-hosted schedule -- no third parallel schedule is added.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ..., description="Workflow correlation ID for this tick."
    )
