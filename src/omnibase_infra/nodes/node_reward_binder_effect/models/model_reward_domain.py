# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Local domain model stubs for reward architecture data models.

These stubs mirror the canonical models from omnibase_core (OMN-2537).
Once OMN-2537 is merged and omnibase_core is released with these models,
these stubs should be replaced with direct imports from omnibase_core.

Ticket: OMN-2552
Dependency: OMN-2537 (core data models — ScoreVector, EvaluationResult, EvidenceBundle)
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ScoreVector(BaseModel):
    """Multi-dimensional score for a single evaluation target.

    Stub pending OMN-2537 merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: UUID = Field(..., description="ID of the target being scored.")
    target_type: Literal["tool", "model", "pattern", "agent"] = Field(
        ..., description="Category of the target."
    )
    dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="Named score dimensions (e.g. accuracy, latency_ms).",
    )
    composite_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Composite scalar in [-1, 1].",
    )


class EvidenceItem(BaseModel):
    """Single unit of evidence supporting an evaluation.

    Stub pending OMN-2537 merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: UUID = Field(default_factory=uuid4, description="Unique ID for this item.")
    source: str = Field(..., description="Source identifier (e.g. 'session_log').")
    content: str = Field(..., description="Raw evidence content.")
    weight: float = Field(default=1.0, ge=0.0, description="Relative weighting.")


class EvidenceBundle(BaseModel):
    """Collection of evidence items for an evaluation run.

    Stub pending OMN-2537 merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_id: UUID = Field(default_factory=uuid4, description="Unique bundle ID.")
    items: tuple[EvidenceItem, ...] = Field(
        default_factory=tuple, description="Evidence items."
    )
    run_id: UUID = Field(..., description="Evaluation run this bundle belongs to.")


class ObjectiveSpec(BaseModel):
    """Specification of an objective used to drive evaluation.

    Stub pending OMN-2537 merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_id: UUID = Field(
        default_factory=uuid4, description="Unique objective ID."
    )
    name: str = Field(..., description="Human-readable objective name.")
    description: str = Field(default="", description="Objective description.")
    target_types: tuple[Literal["tool", "model", "pattern", "agent"], ...] = Field(
        default_factory=tuple,
        description="Target types this objective applies to.",
    )
    weight: float = Field(default=1.0, ge=0.0, description="Relative weight.")


class EvaluationResult(BaseModel):
    """Result produced by ScoringReducer for a single evaluation run.

    Stub pending OMN-2537 merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID = Field(..., description="Unique evaluation run ID.")
    objective_id: UUID = Field(..., description="Objective that drove this run.")
    score_vectors: tuple[ScoreVector, ...] = Field(
        default_factory=tuple,
        description="Per-target score vectors.",
    )
    evidence_bundle: EvidenceBundle = Field(
        ..., description="Evidence supporting this evaluation."
    )
    policy_state_before: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot before this evaluation.",
    )
    policy_state_after: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot after this evaluation.",
    )


__all__: list[str] = [
    "EvaluationResult",
    "EvidenceBundle",
    "EvidenceItem",
    "ObjectiveSpec",
    "ScoreVector",
]
