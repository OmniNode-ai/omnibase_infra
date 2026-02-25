# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Payload model for the write-decision intent.

This payload drives Stage 1 + Stage 2 of HandlerWriteDecision:
    - Stage 1: Normalize, validate, upsert into decision_store
    - Stage 2: Structural conflict detection against active decisions

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
    - OMN-2764: DB migrations
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadWriteDecision(BaseModel):
    """Payload for the decision-store write intent.

    Carries all fields required for Stage 1 (upsert into decision_store)
    and the correlation context needed for Stage 2 conflict detection.

    Attributes:
        intent_type: Routing key — always "decision_store.write_decision".
        correlation_id: Correlation UUID from the originating context.
        decision_id: Caller-supplied UUID for the decision (stable across retries).
        title: Human-readable decision title.
        decision_type: Classification string matching EnumDecisionType values.
        status: Requested lifecycle status ("PROPOSED" or "ACTIVE").
        scope_domain: Domain vocabulary term (validated against ALLOWED_DOMAINS).
        scope_services: Sequence of service/repo slugs affected. Handler normalises
            to a sorted list of lowercase strings before insert.
        scope_layer: Architectural layer ("architecture", "design", "planning",
            "implementation").
        rationale: Human-readable explanation for the decision.
        alternatives: JSON-serialisable list of alternative dicts (each has
            "label", "status", optional "rejection_reason").
        tags: Optional list of free-form tag strings.
        source: Origin of the decision record ("planning", "interview",
            "pr_review", "manual").
        epic_id: Optional parent epic identifier (Linear or ONEX ID).
        supersedes: List of decision_id UUIDs this entry supersedes.
        superseded_by: UUID of the decision that supersedes this one, if any.
            When set, the handler forces status=SUPERSEDED.
        created_at: Application-supplied creation timestamp (timezone-aware).
            Rejected if more than 5 minutes in the future compared to DB now().
        created_by: Identifier of the creator (email, agent ID, etc.).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["decision_store.write_decision"] = Field(
        default="decision_store.write_decision",
        description="Routing key for this intent.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation UUID from the originating pipeline/session.",
    )
    decision_id: UUID = Field(
        ...,
        description="Caller-supplied stable UUID for this decision.",
    )
    title: str = Field(..., description="Human-readable decision title.")
    decision_type: str = Field(
        ...,
        description="Classification matching EnumDecisionType values.",
    )
    status: Literal["PROPOSED", "ACTIVE"] = Field(
        default="ACTIVE",
        description="Requested lifecycle status. Overridden to SUPERSEDED if superseded_by is set.",
    )
    scope_domain: str = Field(
        ...,
        description="Domain vocabulary term. Validated against ALLOWED_DOMAINS by handler.",
    )
    scope_services: list[str] = Field(
        default_factory=list,
        description="Service/repo slugs affected. Normalised (sorted, lowercase) by handler.",
    )
    scope_layer: Literal["architecture", "design", "planning", "implementation"] = (
        Field(
            ...,
            description="Architectural layer this decision affects.",
        )
    )
    rationale: str = Field(
        ..., description="Human-readable explanation for the decision."
    )
    alternatives: list[dict[str, object]] = Field(
        default_factory=list,
        description="Alternative options considered. Each dict has label, status, optional rejection_reason.",
    )
    tags: list[str] = Field(default_factory=list, description="Free-form tag strings.")
    source: Literal["planning", "interview", "pr_review", "manual"] = Field(
        ...,
        description="Origin of this decision record.",
    )
    epic_id: str | None = Field(
        default=None,
        description="Optional parent epic identifier.",
    )
    supersedes: list[UUID] = Field(
        default_factory=list,
        description="UUIDs of decisions this entry supersedes.",
    )
    superseded_by: UUID | None = Field(
        default=None,
        description="UUID of the decision that supersedes this one. Forces status=SUPERSEDED when set.",
    )
    created_at: datetime = Field(
        ...,
        description="Application-supplied creation timestamp. Must be timezone-aware.",
    )
    created_by: str = Field(
        ...,
        description="Creator identifier (email, agent ID, etc.).",
    )


__all__: list[str] = ["ModelPayloadWriteDecision"]
