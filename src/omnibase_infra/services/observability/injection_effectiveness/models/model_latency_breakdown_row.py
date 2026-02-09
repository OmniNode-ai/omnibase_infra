# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Row model for latency_breakdowns table reads.

Represents a single row from the latency_breakdowns table as returned
by query operations.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLatencyBreakdownRow(BaseModel):
    """Single row from the latency_breakdowns table.

    Attributes:
        id: Auto-generated primary key.
        session_id: Session identifier.
        prompt_id: Unique prompt identifier.
        cohort: A/B test cohort.
        cache_hit: Whether prompt benefited from cache.
        routing_latency_ms: Time in agent routing (ms).
        retrieval_latency_ms: Time retrieving context (ms).
        injection_latency_ms: Time injecting context (ms).
        user_latency_ms: User-perceived latency (ms).
        emitted_at: Event time from producer.
        created_at: Ingest timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(..., description="Primary key")
    session_id: UUID = Field(..., description="Session identifier")
    prompt_id: UUID = Field(..., description="Prompt identifier")

    cohort: str | None = Field(default=None, description="A/B test cohort")
    cache_hit: bool = Field(default=False, description="Cache hit flag")

    routing_latency_ms: int | None = Field(default=None, description="Routing latency")
    retrieval_latency_ms: int | None = Field(
        default=None, description="Retrieval latency"
    )
    injection_latency_ms: int | None = Field(
        default=None, description="Injection latency"
    )
    user_latency_ms: int = Field(..., description="User-perceived latency")

    emitted_at: datetime | None = Field(default=None, description="Producer timestamp")
    created_at: datetime = Field(..., description="Ingest timestamp")


__all__ = ["ModelLatencyBreakdownRow"]
