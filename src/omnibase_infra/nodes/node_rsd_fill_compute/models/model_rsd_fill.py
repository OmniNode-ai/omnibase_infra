# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RSD fill compute models.

Related:
    - OMN-7315: node_rsd_fill_compute
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScoredTicket(BaseModel):
    """A ticket with an RSD (Relative Sprint Difficulty) score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str = Field(..., description="Linear ticket identifier.")
    title: str = Field(..., description="Ticket title.")
    rsd_score: float = Field(
        ..., ge=0.0, description="RSD score (higher = more valuable to fill)."
    )
    priority: int = Field(
        default=0, ge=0, le=4, description="Priority (0=none, 1=urgent, 4=low)."
    )
    labels: tuple[str, ...] = Field(
        default_factory=tuple, description="Ticket labels."
    )
    description: str = Field(default="", description="Ticket description.")
    state: str = Field(default="", description="Current ticket state.")


class ModelRsdFillInput(BaseModel):
    """Input to the RSD fill compute node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Build loop cycle correlation ID."
    )
    scored_tickets: tuple[ModelScoredTicket, ...] = Field(
        ..., description="All scored tickets available for selection."
    )
    max_tickets: int = Field(
        default=5, ge=1, le=20, description="Maximum tickets to select."
    )


class ModelRsdFillOutput(BaseModel):
    """Output from the RSD fill compute node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Build loop cycle correlation ID."
    )
    selected_tickets: tuple[ModelScoredTicket, ...] = Field(
        ..., description="Top-N tickets selected by RSD score."
    )
    total_candidates: int = Field(
        ..., ge=0, description="Total candidates considered."
    )
    total_selected: int = Field(
        ..., ge=0, description="Number of tickets selected."
    )


__all__: list[str] = [
    "ModelRsdFillInput",
    "ModelRsdFillOutput",
    "ModelScoredTicket",
]
