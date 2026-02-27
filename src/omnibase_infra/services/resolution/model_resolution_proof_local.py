# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Local resolution proof model for resolution event ledger.

Mirrors the essential fields from ``ModelResolutionProof`` in omnibase_core.

TODO(OMN-2895): Replace with the canonical import once omnibase_core PR #575
    merges and is available as a dependency.

Related:
    - OMN-2895: Resolution Event Ledger (Phase 6 of OMN-2897 epic)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelResolutionProofLocal(BaseModel):
    """Local representation of a resolution proof attempt.

    Mirrors the essential fields from ``ModelResolutionProof`` in omnibase_core.
    Records whether a specific proof type was verified during resolution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    proof_type: str = Field(..., description="Type of proof attempted")
    verified: bool = Field(default=False, description="Whether the proof was verified")
    verification_notes: str = Field(
        default="", description="Additional notes on verification"
    )
    verified_at: datetime | None = Field(
        default=None, description="When verification completed"
    )


__all__: list[str] = ["ModelResolutionProofLocal"]
