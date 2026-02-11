# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Phase payload for the pr_release_ready phase.

Records last review SHA and issue fingerprints after PR release-ready review.

Ticket: OMN-2143
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPhasePayloadPrReleaseReady(BaseModel):
    """Payload captured after the pr_release_ready phase completes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    phase: Literal["pr_release_ready"] = Field(
        default="pr_release_ready",
        description="Discriminator field for phase payload union.",
    )
    last_review_sha: str = Field(
        ...,
        min_length=7,
        max_length=40,
        description="Commit SHA of the last review pass.",
    )
    issue_fingerprints: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Fingerprints of issues found and resolved.",
    )


__all__: list[str] = ["ModelPhasePayloadPrReleaseReady"]
