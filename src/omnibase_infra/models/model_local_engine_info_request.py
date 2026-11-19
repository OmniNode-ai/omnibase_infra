# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the local provision node's ``engine_info`` read (OMN-23413)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.operational_timeouts import TIMEOUTS


class ModelLocalEngineInfoRequest(BaseModel):
    """One read of the Docker engine's CPU count and memory."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Correlation ID for tracing.")
    timeout_seconds: float = Field(
        default=TIMEOUTS.dev_lane_docker_seconds,
        gt=0,
        allow_inf_nan=False,
        description="Maximum wait for one Docker engine observation, in seconds.",
    )


__all__: list[str] = ["ModelLocalEngineInfoRequest"]
