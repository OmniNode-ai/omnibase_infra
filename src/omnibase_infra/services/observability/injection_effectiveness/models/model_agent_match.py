# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent match event model.

Represents the payload from onex.evt.omniclaude.agent-match.v1 topic.
Tracks agent routing accuracy with graded scoring (replaces boolean match).

Related Tickets:
    - OMN-1889: Emit injection metrics + utilization signal (producer)
    - OMN-1890: Store injection metrics with corrected schema (consumer)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelAgentMatchEvent(BaseModel):
    """Agent match event from omniclaude hooks.

    Emitted to report how well the predicted agent matched the actual
    agent used in the session. Uses graded scoring (0.0-1.0) instead
    of simple boolean match for more nuanced analysis.

    This event populates:
        - injection_effectiveness table (agent match fields)

    Scoring Logic:
        - 1.0: Exact match (expected == actual)
        - 0.5-0.9: Partial match (same category, different variant)
        - 0.0: Complete mismatch (different category)

    Attributes:
        event_type: Event type discriminator.
        session_id: Session identifier for correlation.
        correlation_id: Request correlation ID for tracing.
        agent_match_score: Graded match score (0.0-1.0).
        expected_agent: Agent predicted by routing system.
        actual_agent: Agent actually used in session.
        created_at: Event timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: Literal["agent_match"] = Field(
        default="agent_match",
        description="Event type discriminator",
    )
    session_id: UUID = Field(..., description="Session identifier")
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )

    # Agent matching (graded 0.0-1.0)
    agent_match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Graded match score 0.0-1.0",
    )
    expected_agent: str | None = Field(
        default=None,
        description="Agent predicted by routing system",
    )
    actual_agent: str | None = Field(
        default=None,
        description="Agent actually used in session",
    )

    # Timestamp
    created_at: datetime = Field(
        ...,
        description="Event timestamp",
    )
