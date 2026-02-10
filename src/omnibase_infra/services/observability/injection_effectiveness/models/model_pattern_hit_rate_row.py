# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Row model for pattern_hit_rates table reads.

Represents a single row from the pattern_hit_rates table as returned
by query operations.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPatternHitRateRow(BaseModel):
    """Single row from the pattern_hit_rates table.

    Attributes:
        id: Auto-generated primary key.
        pattern_id: Pattern identifier.
        domain_id: Domain scope (optional).
        utilization_method: Method used to measure utilization.
        utilization_score: Rolling average score (0.0-1.0).
        hit_count: Times pattern was used in response.
        miss_count: Times pattern was injected but not used.
        sample_count: Total observations.
        confidence: Confidence level (None if sample_count < 20).
        created_at: Row creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(..., description="Primary key")
    pattern_id: UUID = Field(..., description="Pattern identifier")
    domain_id: UUID | None = Field(default=None, description="Domain scope")

    utilization_method: str = Field(..., description="Measurement method")
    utilization_score: float = Field(..., description="Rolling average score 0.0-1.0")
    hit_count: int = Field(default=0, description="Hit count")
    miss_count: int = Field(default=0, description="Miss count")
    sample_count: int = Field(default=0, description="Total observations")
    confidence: float | None = Field(
        default=None, description="Confidence (None if N<20)"
    )

    created_at: datetime = Field(..., description="Row creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


__all__ = ["ModelPatternHitRateRow"]
