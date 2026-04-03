# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injection recorded event model.

Represents the payload from onex.evt.omniclaude.injection-recorded.v1 topic.
Tracks context injection tracking events emitted by INJECT-004 (OMN-1673).

Producer: omniclaude context injection hooks (OMN-1673)
Consumer: injection_effectiveness consumer (OMN-6158)

Related Tickets:
    - OMN-1673: INJECT-004 injection tracking event emission (producer)
    - OMN-6158: Add consumer for injection.recorded events (this)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelInjectionRecordedEvent(BaseModel):
    """Injection recorded event from omniclaude hooks.

    Emitted when a context injection is recorded during a session.
    Tracks what was injected, how many tokens, and which patterns were used.

    This event populates the ``injection_recorded_events`` table.

    Attributes:
        session_id: Session identifier for correlation.
        correlation_id: Request correlation ID for tracing.
        emitted_at: Timestamp when the hook emitted this event (UTC).
        patterns_injected: Number of patterns injected.
        total_injected_tokens: Total tokens injected into context.
        injection_latency_ms: Time to perform the injection in milliseconds.
        agent_name: Agent that triggered the injection.
        repo: Repository name derived from project_path.
        cache_hit: Whether the injection result was cached.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    session_id: str = Field(..., description="Session identifier")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Correlation ID for tracing",
    )
    emitted_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp when the hook emitted this event",
    )

    # Injection metrics
    patterns_injected: int = Field(
        default=0,
        ge=0,
        description="Number of patterns injected",
    )
    total_injected_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens injected into context",
    )
    injection_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time to perform the injection in milliseconds",
    )

    # Context
    agent_name: str | None = Field(
        default=None,
        description="Agent that triggered the injection",
    )
    repo: str | None = Field(
        default=None,
        description="Repository name derived from project_path",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether the injection result was cached",
    )
