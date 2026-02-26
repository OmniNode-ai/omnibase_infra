# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Local tier attempt model for resolution event ledger.

Mirrors the essential fields from ``ModelTierAttempt`` in omnibase_core.

TODO(OMN-2895): Replace with the canonical import once omnibase_core PR #575
    merges and is available as a dependency.

Related:
    - OMN-2895: Resolution Event Ledger (Phase 6 of OMN-2897 epic)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelTierAttemptLocal(BaseModel):
    """Local representation of a tier attempt within resolution progression.

    Mirrors the essential fields from ``ModelTierAttempt`` in omnibase_core.
    Records the outcome of attempting resolution at a single tier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    tier: str = Field(..., description="Resolution tier attempted")
    attempted_at: datetime = Field(..., description="When this tier attempt started")
    candidates_found: int = Field(
        default=0, description="Number of candidates found at this tier"
    )
    candidates_after_trust_filter: int = Field(
        default=0, description="Candidates remaining after trust filtering"
    )
    failure_code: str | None = Field(
        default=None, description="Structured failure code if tier failed"
    )
    failure_reason: str | None = Field(
        default=None, description="Human-readable failure reason"
    )
    duration_ms: float = Field(
        default=0.0, description="Duration of this tier attempt in milliseconds"
    )


__all__: list[str] = ["ModelTierAttemptLocal"]
