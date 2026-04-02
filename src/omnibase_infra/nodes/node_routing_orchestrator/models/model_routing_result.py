# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Final routing result emitted by the orchestrator."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRoutingResult(BaseModel):
    """Final result of the routing workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    selected_model_key: str = Field(..., description="Selected model ID.")
    selected_endpoint_env: str = Field(
        ..., description="Env var for the selected endpoint."
    )
    fallback_model_key: str | None = Field(
        default=None, description="Fallback model ID."
    )
    rationale: str = Field(..., description="Routing rationale.")
    estimated_cost: float = Field(default=0.0, description="Estimated cost USD.")
    estimated_latency_ms: int = Field(default=0, description="Estimated latency ms.")
    success: bool = Field(default=True, description="Whether routing succeeded.")
    error_message: str = Field(default="", description="Error if failed.")
