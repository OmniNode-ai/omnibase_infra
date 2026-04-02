# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hard constraints that filter candidate models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRoutingConstraints(BaseModel):
    """Hard constraints that filter candidate models."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    max_cost_per_1k: float = Field(
        default=0.10, description="Max $/1K tokens (0 = local only)."
    )
    min_quality_score: float = Field(
        default=0.0, description="Minimum quality score 0-1."
    )
    max_latency_ms: int = Field(
        default=30000, description="Max acceptable first-token latency in ms."
    )
    needs_vision: bool = Field(default=False, description="Requires vision capability.")
    needs_computer_use: bool = Field(
        default=False, description="Requires computer use capability."
    )
    needs_tool_use: bool = Field(
        default=False, description="Requires tool use capability."
    )
    min_context_window: int = Field(
        default=4096, description="Minimum context window in tokens."
    )
    prefer_local: bool = Field(
        default=True, description="Tie-break toward local models."
    )
