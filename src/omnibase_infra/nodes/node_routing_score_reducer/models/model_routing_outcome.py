# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Routing outcome event — fed back to the reducer after task completion."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_router_compute.models.model_routing_request import (
    EnumTaskType,
)


class ModelRoutingOutcome(BaseModel):
    """Outcome of a routing decision — success/failure + metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    model_key: str = Field(..., description="Model that was used.")
    task_type: EnumTaskType = Field(..., description="Task type that was routed.")
    task_subtype: str = Field(default="", description="Optional sub-classification.")
    success: bool = Field(..., description="Whether the task succeeded.")
    actual_latency_ms: int = Field(
        default=0, description="Observed end-to-end latency in ms."
    )
    actual_tokens_per_sec: float = Field(
        default=0.0, description="Observed throughput."
    )
    actual_cost: float = Field(default=0.0, description="Actual cost in USD.")
    input_tokens: int = Field(default=0, description="Input token count.")
    output_tokens: int = Field(default=0, description="Output token count.")
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="When the task completed.",
    )
