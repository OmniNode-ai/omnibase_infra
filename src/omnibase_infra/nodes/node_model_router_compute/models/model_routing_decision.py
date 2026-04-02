# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Routing decision output model."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRoutingDecision(BaseModel):
    """Output of the model router compute node — the routing decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    selected_model_key: str = Field(..., description="ID of the selected model.")
    selected_endpoint_env: str = Field(
        ...,
        description="Env var name for the selected model endpoint (e.g. LLM_CODER_URL).",
    )
    fallback_model_key: str | None = Field(
        default=None, description="Fallback model if primary fails."
    )
    rationale: str = Field(
        ..., description="Human-readable explanation of the routing decision."
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Model ID -> composite score mapping.",
    )
    estimated_cost: float = Field(
        default=0.0, description="Estimated cost in USD for this request."
    )
    estimated_latency_ms: int = Field(
        default=0, description="Estimated latency in milliseconds."
    )
    success: bool = Field(default=True, description="Whether routing succeeded.")
    error_message: str = Field(
        default="", description="Error message if routing failed."
    )
