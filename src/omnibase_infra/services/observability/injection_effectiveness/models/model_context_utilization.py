# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Context utilization event model.

Represents the payload from onex.evt.omniclaude.context-utilization.v1 topic.
Tracks whether injected context was actually used in the Claude response.

Related Tickets:
    - OMN-1889: Emit injection metrics + utilization signal (producer)
    - OMN-1890: Store injection metrics with corrected schema (consumer)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.observability.injection_effectiveness.models.model_pattern_utilization import (
    ModelPatternUtilization,
)


class ModelContextUtilizationEvent(BaseModel):
    """Context utilization event from omniclaude hooks.

    Emitted at session end to report how much of the injected context
    was actually utilized in Claude's responses.

    This event populates:
        - injection_effectiveness table (utilization fields)
        - pattern_hit_rates table (per-pattern breakdown)

    Attributes:
        event_type: Event type discriminator.
        session_id: Session identifier for correlation.
        correlation_id: Request correlation ID for tracing.
        cohort: A/B test cohort assignment.
        cohort_identity_type: How cohort was assigned.
        total_injected_tokens: Total tokens injected in session.
        patterns_injected: Number of patterns injected.
        utilization_score: Overall utilization score (0.0-1.0).
        utilization_method: Detection method used.
        injected_identifiers_count: Identifiers in injected context.
        reused_identifiers_count: Identifiers found in response.
        pattern_utilizations: Per-pattern utilization breakdown.
        created_at: Event timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: Literal["context_utilization"] = Field(
        default="context_utilization",
        description="Event type discriminator",
    )
    session_id: UUID = Field(..., description="Session identifier")
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )

    # A/B Testing
    cohort: Literal["control", "treatment"] | None = Field(
        default=None,
        description="A/B test cohort",
    )
    cohort_identity_type: Literal["user_id", "repo_path", "session_id"] | None = Field(
        default=None,
        description="Identity type used for cohort assignment",
    )

    # Injection metrics
    total_injected_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens injected in session",
    )
    patterns_injected: int = Field(
        default=0,
        ge=0,
        description="Number of patterns injected",
    )

    # Utilization (heuristic-based, NOT token attribution)
    utilization_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall utilization score 0.0-1.0",
    )
    utilization_method: str = Field(
        ...,
        description="Detection method: identifier_match, semantic, or timeout",
    )
    injected_identifiers_count: int = Field(
        default=0,
        ge=0,
        description="Number of identifiers in injected context",
    )
    reused_identifiers_count: int = Field(
        default=0,
        ge=0,
        description="Number of identifiers found in response",
    )

    # Per-pattern breakdown
    pattern_utilizations: list[ModelPatternUtilization] = Field(
        default_factory=list,
        description="Per-pattern utilization metrics",
    )

    # Timestamp
    created_at: datetime = Field(
        ...,
        description="Event timestamp",
    )
