# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the local provision node's ``engine_info`` read (OMN-23413)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLocalEngineInfoRequest(BaseModel):
    """One read of the Docker engine's CPU count and memory."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Correlation ID for tracing.")


__all__: list[str] = ["ModelLocalEngineInfoRequest"]
