# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Pydantic model for routing-decided observability events.

Emitted by AdapterModelRouter after each routing decision. Published to
Kafka via the RoutingEventCallback and consumed by observability tooling.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRoutingDecidedEvent(BaseModel):
    """Schema for a routing-decided event emitted by AdapterModelRouter.

    Fields
    ------
    selection_mode:
        How the winning provider was selected. One of:

        - ``"round_robin"`` — normal load-balanced pick, no prior failures
        - ``"failover"`` — a previous provider failed; fell back to this one
        - ``"priority"`` — selected by explicit priority rank (future)
        - ``"cost_optimized"`` — selected by cost scoring (future)

    fallback_indicator:
        ``True`` when the selected provider was **not** the first candidate
        tried in this call — i.e., at least one provider was skipped or
        failed before this provider succeeded.  Mirrors ``is_fallback`` but
        uses the canonical SOW field name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID | None = Field(
        default=None, description="Request correlation ID"
    )
    session_id: str | None = Field(default=None, description="Optional session ID")

    selected_provider: str = Field(..., description="Name of the chosen provider")
    selected_tier: str = Field(
        ..., description="Tier of the chosen provider: local, cheap_cloud, or claude"
    )
    selected_model: str = Field(default="", description="Model ID from the request")

    selection_mode: Literal[
        "round_robin",
        "failover",
        "priority",
        "cost_optimized",
    ] = Field(
        ...,
        description=(
            "How the provider was selected. 'failover' when one or more providers "
            "were attempted and failed before this one succeeded; 'round_robin' "
            "for a normal first-try pick."
        ),
    )
    fallback_indicator: bool = Field(
        ...,
        description=(
            "True when the selected provider was not the first candidate attempted "
            "in this call (i.e., at least one provider failed before this one)."
        ),
    )

    # Kept for backwards compatibility with existing consumers that read is_fallback.
    is_fallback: bool = Field(
        ...,
        description="Deprecated alias for fallback_indicator. Use fallback_indicator.",
    )

    reason: str = Field(
        ...,
        description="Human-readable reason string: 'fallback' or 'round_robin'",
    )
    candidates_evaluated: int = Field(
        ..., description="Number of providers examined before selecting this one"
    )
    candidate_providers: list[str] = Field(
        default_factory=list,
        description="Full ordered list of registered providers at decision time",
    )
    task_type: str | None = Field(
        default=None, description="Optional task type from the request"
    )
    latency_ms: float = Field(..., description="Wall-clock ms from call start to pick")
    timestamp: str = Field(
        ..., description="ISO-8601 UTC timestamp of the routing decision"
    )


__all__: list[str] = ["ModelRoutingDecidedEvent"]
