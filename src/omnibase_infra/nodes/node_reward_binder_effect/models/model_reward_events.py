# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Kafka event models emitted by the RewardBinder EFFECT node.

Three event types are emitted in order per evaluation run:
  1. RunEvaluatedEvent  — overall run summary with objective fingerprint
  2. RewardAssignedEvent — per-target reward with traceable evidence refs
  3. PolicyStateUpdatedEvent — policy state transition with old/new snapshots

Ticket: OMN-2552
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelRunEvaluatedEvent(BaseModel):
    """Event emitted after every evaluation run.

    Published to: ``{env}.onex.evt.omnimemory.run-evaluated.v1``

    The ``objective_fingerprint`` is a SHA-256 hex digest of the serialized
    ``ObjectiveSpec`` used for the run — providing tamper-evident traceability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier.",
    )
    run_id: UUID = Field(
        ...,
        description="Evaluation run ID from EvaluationResult.",
    )
    objective_id: UUID = Field(
        ...,
        description="Objective that drove this evaluation.",
    )
    objective_fingerprint: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of ObjectiveSpec.model_dump_json() — tamper-evident.",
    )
    composite_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of target_id (str) -> composite_score for quick access.",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event emission.",
    )


class ModelRewardAssignedEvent(BaseModel):
    """Event emitted for each target_type that received a reward.

    Published to: ``{env}.onex.evt.omnimemory.reward-assigned.v1``

    ``evidence_refs`` must be traceable back to specific ``EvidenceItem.item_id``
    values from the input ``EvidenceBundle``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier.",
    )
    run_id: UUID = Field(
        ...,
        description="Evaluation run ID.",
    )
    target_id: UUID = Field(
        ...,
        description="ID of the target receiving this reward.",
    )
    target_type: Literal["tool", "model", "pattern", "agent"] = Field(
        ...,
        description="Category of the reward target.",
    )
    composite_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Composite scalar reward in [-1, 1].",
    )
    dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="Named dimension scores from ScoreVector.",
    )
    evidence_refs: tuple[UUID, ...] = Field(
        default_factory=tuple,
        description="EvidenceItem.item_id values supporting this reward.",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event emission.",
    )


class ModelPolicyStateUpdatedEvent(BaseModel):
    """Event emitted when policy state transitions.

    Published to: ``{env}.onex.evt.omnimemory.policy-state-updated.v1``

    Includes both ``old_state`` and ``new_state`` snapshots for auditability.
    Only emitted when the policy state actually changed between evaluations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier.",
    )
    run_id: UUID = Field(
        ...,
        description="Evaluation run ID that triggered the state transition.",
    )
    old_state: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot before the evaluation.",
    )
    new_state: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot after the evaluation.",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event emission.",
    )


__all__: list[str] = [
    "ModelPolicyStateUpdatedEvent",
    "ModelRewardAssignedEvent",
    "ModelRunEvaluatedEvent",
]
