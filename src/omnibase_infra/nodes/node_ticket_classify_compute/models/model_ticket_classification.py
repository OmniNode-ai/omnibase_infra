# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ticket classification model for buildability assessment.

Related:
    - OMN-7312: ModelTicketClassification
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_buildability import EnumBuildability


class ModelTicketClassification(BaseModel):
    """Classification result for a single ticket.

    Produced by the ticket classify compute node using keyword heuristics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str = Field(
        ..., description="Linear ticket identifier (e.g. OMN-1234)."
    )
    title: str = Field(
        ..., description="Ticket title."
    )
    buildability: EnumBuildability = Field(
        ..., description="Buildability classification."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score."
    )
    matched_keywords: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Keywords that contributed to the classification.",
    )
    reason: str = Field(
        default="", description="Human-readable classification rationale."
    )


class ModelTicketClassifyInput(BaseModel):
    """Input to the ticket classify compute node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Build loop cycle correlation ID."
    )
    tickets: tuple[ModelTicketForClassification, ...] = Field(
        ..., description="Tickets to classify."
    )


class ModelTicketForClassification(BaseModel):
    """A single ticket to be classified."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket_id: str = Field(..., description="Linear ticket identifier.")
    title: str = Field(..., description="Ticket title.")
    description: str = Field(default="", description="Ticket description/body.")
    labels: tuple[str, ...] = Field(
        default_factory=tuple, description="Ticket labels."
    )
    state: str = Field(default="", description="Current ticket state.")
    priority: int = Field(default=0, ge=0, le=4, description="Priority (0=none, 1=urgent, 4=low).")


class ModelTicketClassifyOutput(BaseModel):
    """Output from the ticket classify compute node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        ..., description="Build loop cycle correlation ID."
    )
    classifications: tuple[ModelTicketClassification, ...] = Field(
        ..., description="Classification results."
    )
    total_auto_buildable: int = Field(
        default=0, ge=0, description="Count of AUTO_BUILDABLE tickets."
    )
    total_skipped: int = Field(
        default=0, ge=0, description="Count of SKIP + BLOCKED + NEEDS_ARCH_DECISION tickets."
    )


__all__: list[str] = [
    "ModelTicketClassification",
    "ModelTicketClassifyInput",
    "ModelTicketClassifyOutput",
    "ModelTicketForClassification",
]
