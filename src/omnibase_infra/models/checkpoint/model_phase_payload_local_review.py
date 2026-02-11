# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Phase payload for the local_review phase.

Records iteration count, issue fingerprints, and the last clean commit SHA.

Ticket: OMN-2143
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPhasePayloadLocalReview(BaseModel):
    """Payload captured after the local_review phase completes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    phase: Literal["local_review"] = Field(
        default="local_review",
        description="Discriminator field for phase payload union.",
    )
    iteration_count: int = Field(
        ...,
        ge=1,
        description="Number of review-fix iterations performed.",
    )
    issue_fingerprints: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Fingerprints of issues found and resolved.",
    )
    last_clean_sha: str = Field(
        ...,
        min_length=7,
        max_length=40,
        pattern=r"^[0-9a-f]+$",
        description="Commit SHA of the last clean (review-passing) state.",
    )


__all__: list[str] = ["ModelPhasePayloadLocalReview"]
