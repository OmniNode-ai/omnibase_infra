# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Context enrichment event model.

Represents the payload from onex.evt.omniclaude.context-enrichment.v1 topic.
Tracks per-channel enrichment outcomes (summarization, code_analysis, similarity)
with latency, token delta, and quality metrics.

Producer: omniclaude enrichment_observability_emitter.py (OMN-2274)
Consumer: injection_effectiveness consumer (OMN-6158)

Related Tickets:
    - OMN-2274: Enrichment observability event emission (producer)
    - OMN-2441: Canonical omnidash ContextEnrichmentEvent schema
    - OMN-6158: Add consumer for context-enrichment events (this)
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelContextEnrichmentEvent(BaseModel):
    """Context enrichment event from omniclaude hooks.

    Emitted per enrichment channel after the enrichment pipeline completes.
    One event per channel (summarization, code_analysis, similarity).

    This event populates the ``context_enrichment_events`` table.

    Attributes:
        session_id: Session identifier for correlation.
        correlation_id: Request correlation ID for tracing.
        timestamp: ISO-8601 UTC timestamp of emission.
        channel: Enrichment channel (summarization, code_analysis, similarity).
        model_name: Model identifier used by the handler.
        cache_hit: Whether the enrichment result was cached.
        outcome: Enrichment outcome (hit, miss, error, inflated).
        latency_ms: Wall-clock duration of the enrichment in milliseconds.
        tokens_before: Token count of the original prompt (pre-enrichment).
        tokens_after: Token count of the produced markdown (0 on failure).
        net_tokens_saved: tokens_before - tokens_after (summarization only).
        similarity_score: Optional float [0.0, 1.0] from handler.
        quality_score: Optional quality score (not yet implemented).
        repo: Repository name derived from project_path.
        agent_name: Agent that triggered the enrichment.
    """

    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    session_id: UUID = Field(..., description="Session identifier")
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing",
    )
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of emission")

    # Channel identification
    channel: str = Field(
        ...,
        description="Enrichment channel: summarization, code_analysis, similarity",
    )
    model_name: str = Field(
        default="",
        description="Model identifier used by the handler",
    )

    # Outcome
    cache_hit: bool = Field(
        default=False,
        description="Whether the enrichment result was cached",
    )
    outcome: str = Field(
        ...,
        description="Enrichment outcome: hit, miss, error, inflated",
    )

    # Metrics
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of the enrichment in milliseconds",
    )
    tokens_before: int = Field(
        default=0,
        ge=0,
        description="Token count of the original prompt (pre-enrichment)",
    )
    tokens_after: int = Field(
        default=0,
        ge=0,
        description="Token count of the produced markdown (0 on failure)",
    )
    net_tokens_saved: int = Field(
        default=0,
        description="tokens_before - tokens_after (summarization channel only)",
    )

    # Scores
    similarity_score: float | None = Field(
        default=None,
        description="Optional float [0.0, 1.0] from handler (similarity channel)",
    )
    quality_score: float | None = Field(
        default=None,
        description="Optional quality score (not yet implemented)",
    )

    # Context
    repo: str | None = Field(
        default=None,
        description="Repository name derived from project_path",
    )
    agent_name: str | None = Field(
        default=None,
        description="Agent that triggered the enrichment",
    )
