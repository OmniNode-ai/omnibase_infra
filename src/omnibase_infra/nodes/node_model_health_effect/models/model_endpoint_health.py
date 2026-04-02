# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Health status of a single model endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEndpointHealth(BaseModel):
    """Health status of a single model endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model_key: str = Field(..., description="Model identifier from registry.")
    healthy: bool = Field(
        ..., description="Whether the endpoint responded successfully."
    )
    latency_ms: int = Field(default=0, description="Response latency in milliseconds.")
    queue_depth: int = Field(
        default=0,
        description="Number of pending requests (from vLLM /metrics if available).",
    )
    error_message: str = Field(default="", description="Error message if probe failed.")
