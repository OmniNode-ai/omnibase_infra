# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the In Progress demotion leg (OMN-18489)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelInProgressDemotionRequest(BaseModel):
    """Inputs of one demotion pass. Every bound arrives from the sweep contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Run correlation ID.")
    window_hours: int = Field(
        ...,
        gt=0,
        description=(
            "Idle window. A ticket is a demotion candidate only when it entered "
            "In Progress, and received a non-sweep comment, no more recently "
            "than this many hours ago. Required: the window has no default."
        ),
    )
    apply: bool = Field(
        ...,
        description="False reports every decision as WOULD_DEMOTE and writes nothing.",
    )
    max_demotions_per_run: int = Field(
        ...,
        ge=0,
        description="Hard cap on state writes in one pass; 0 writes none.",
    )
    pr_owner: str = Field(
        ...,
        min_length=1,
        description="GitHub organisation whose open pull requests are searched for the ticket id.",
    )
    gh_timeout_seconds: int = Field(..., gt=0)
    exclude_tickets: tuple[str, ...] = Field(
        default=(),
        description="The exclusion fence: tickets this pass must not touch.",
    )


__all__ = ["ModelInProgressDemotionRequest"]
