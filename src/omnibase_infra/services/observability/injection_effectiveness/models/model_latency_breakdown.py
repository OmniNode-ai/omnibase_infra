# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Latency breakdown event model.

Represents the payload from onex.evt.omniclaude.latency-breakdown.v1 topic.
Tracks detailed timing including user-perceived latency per prompt.

Related Tickets:
    - OMN-1889: Emit injection metrics + utilization signal (producer)
    - OMN-1890: Store injection metrics with corrected schema (consumer)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLatencyBreakdownEvent(BaseModel):
    """Latency breakdown event from omniclaude hooks.

    Emitted per-prompt to report detailed latency breakdowns including
    internal timings and user-perceived latency.

    This event populates:
        - latency_breakdowns table (per-prompt row)
        - injection_effectiveness table (MAX aggregation for user_visible_latency_ms)

    Attributes:
        event_type: Event type discriminator.
        session_id: Session identifier for correlation.
        correlation_id: Request correlation ID for tracing.
        prompt_id: Unique prompt identifier from emitter.
        cohort: A/B test cohort assignment.
        cache_hit: Whether this prompt benefited from cache.
        routing_latency_ms: Time spent in agent routing.
        retrieval_latency_ms: Time spent retrieving context.
        injection_latency_ms: Time spent injecting context.
        user_latency_ms: User-perceived latency for this prompt.
        emitted_at: Event timestamp from producer.
        created_at: Event timestamp (ingest time).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: Literal["latency_breakdown"] = Field(
        default="latency_breakdown",
        description="Event type discriminator",
    )
    session_id: UUID = Field(..., description="Session identifier")
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )

    # Prompt identification (UUID instead of index for uniqueness)
    prompt_id: UUID = Field(
        ...,
        description="Unique prompt identifier from emitter",
    )

    # A/B Testing (denormalized for fast queries)
    cohort: Literal["control", "treatment"] | None = Field(
        default=None,
        description="A/B test cohort",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether this prompt benefited from cached context",
    )

    # Internal timings (ms) - nullable as not all prompts have all timings
    routing_latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Time spent in agent routing (ms)",
    )
    retrieval_latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Time spent retrieving context (ms)",
    )
    injection_latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Time spent injecting context (ms)",
    )

    # User-perceived latency (required)
    user_latency_ms: int = Field(
        ...,
        ge=0,
        description="User-perceived latency for this prompt (ms)",
    )

    # Timestamps
    emitted_at: datetime | None = Field(
        default=None,
        description="Event timestamp from producer (for drift analysis)",
    )
    created_at: datetime = Field(
        ...,
        description="Event timestamp (ingest time)",
    )
