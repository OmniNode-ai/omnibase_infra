# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Row model for injection_effectiveness table reads.

Represents a single row from the injection_effectiveness table as returned
by query operations. All nullable columns map to Optional fields.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
    - OMN-1890: Store injection metrics with corrected schema
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelInjectionEffectivenessRow(BaseModel):
    """Single row from the injection_effectiveness table.

    Attributes:
        session_id: Session identifier (primary key).
        correlation_id: Request correlation ID for tracing.
        realm: Trust domain / environment.
        runtime_id: Runtime instance identifier.
        routing_path: Event routing path.
        cohort: A/B test cohort assignment.
        cohort_identity_type: How cohort was assigned.
        total_injected_tokens: Total tokens injected in session.
        patterns_injected: Number of patterns injected.
        utilization_score: Overall utilization score (0.0-1.0).
        utilization_method: Detection method used.
        injected_identifiers_count: Identifiers in injected context.
        reused_identifiers_count: Identifiers found in response.
        agent_match_score: Graded agent match score (0.0-1.0).
        expected_agent: Agent predicted by routing system.
        actual_agent: Agent actually used.
        user_visible_latency_ms: MAX of prompt latencies.
        created_at: Row creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    session_id: UUID = Field(..., description="Session identifier (PK)")
    correlation_id: UUID | None = Field(default=None, description="Correlation ID")

    # Environment context
    realm: str | None = Field(default=None, description="Trust domain")
    runtime_id: str | None = Field(default=None, description="Runtime instance ID")
    routing_path: str | None = Field(default=None, description="Routing path")

    # A/B testing
    cohort: str | None = Field(default=None, description="A/B test cohort")
    cohort_identity_type: str | None = Field(
        default=None, description="Cohort identity type"
    )

    # Injection metrics
    total_injected_tokens: int | None = Field(
        default=None, description="Total tokens injected"
    )
    patterns_injected: int | None = Field(
        default=None, description="Number of patterns injected"
    )

    # Utilization
    utilization_score: float | None = Field(
        default=None, description="Utilization score 0.0-1.0"
    )
    utilization_method: str | None = Field(
        default=None, description="Utilization detection method"
    )
    injected_identifiers_count: int | None = Field(
        default=None, description="Injected identifiers count"
    )
    reused_identifiers_count: int | None = Field(
        default=None, description="Reused identifiers count"
    )

    # Agent matching
    agent_match_score: float | None = Field(
        default=None, description="Agent match score 0.0-1.0"
    )
    expected_agent: str | None = Field(default=None, description="Predicted agent")
    actual_agent: str | None = Field(default=None, description="Actual agent")

    # Latency
    user_visible_latency_ms: int | None = Field(
        default=None, description="MAX of prompt latencies"
    )

    # Timestamps
    created_at: datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


__all__ = ["ModelInjectionEffectivenessRow"]
