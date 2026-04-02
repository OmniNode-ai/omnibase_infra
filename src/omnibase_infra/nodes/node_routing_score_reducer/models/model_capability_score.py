# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rolling capability score for a (model_key, task_type) pair."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)


class ModelCapabilityScore(BaseModel):
    """Rolling capability score for a (model_key, task_type) pair."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model_key: str = Field(..., description="Model identifier.")
    task_type: EnumTaskType = Field(..., description="Task type.")
    success_count: int = Field(default=0, description="Successes in rolling window.")
    failure_count: int = Field(default=0, description="Failures in rolling window.")
    total_count: int = Field(default=0, description="Total outcomes in window.")
    success_rate: float = Field(default=0.0, description="Rolling success rate 0-1.")
    avg_latency_ms: int = Field(default=0, description="Average latency in ms.")
    avg_tokens_per_sec: float = Field(default=0.0, description="Average throughput.")
    total_cost: float = Field(default=0.0, description="Cumulative cost in USD.")
    graduated: bool = Field(
        default=False,
        description="Whether model graduated for this task type (>0.9 over 50+ attempts).",
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC),
        description="Last update timestamp.",
    )
