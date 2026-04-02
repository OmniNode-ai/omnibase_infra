# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live observed metrics for a (model_key, task_type) pair from the reducer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)


class ModelLiveMetrics(BaseModel):
    """Live observed metrics for a (model_key, task_type) pair from the reducer."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model_key: str = Field(..., description="Model identifier.")
    task_type: EnumTaskType = Field(..., description="Task type these metrics are for.")
    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Rolling success rate 0-1."
    )
    sample_count: int = Field(
        default=0, ge=0, description="Number of verified outcomes."
    )
    avg_latency_ms: int = Field(
        default=0, ge=0, description="Average observed latency in ms."
    )
    avg_tokens_per_sec: float = Field(
        default=0.0, ge=0.0, description="Average observed throughput."
    )
    graduated: bool = Field(
        default=False,
        description="Whether this model has graduated for this task type.",
    )
