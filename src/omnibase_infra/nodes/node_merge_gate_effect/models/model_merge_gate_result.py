# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Payload model for merge gate decision events.

Deserialized from ``onex.evt.platform.merge-gate-decision.v1`` Kafka events.
Carries all fields needed for upserting into the ``merge_gate_decisions`` table
and for opening Linear quarantine tickets when decision == QUARANTINE.

Related Tickets:
    - OMN-3140: NodeMergeGateEffect + migration
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_gate_effect.models.model_merge_gate_violation import (
    ModelMergeGateViolation,
)


class ModelMergeGateResult(BaseModel):
    """Payload for the merge gate decision event.

    Deserialized from ``onex.evt.platform.merge-gate-decision.v1``.

    Attributes:
        gate_id: Unique identifier for this gate evaluation.
        pr_ref: Pull request reference (e.g. "OmniNode-ai/omnibase_infra#42").
        head_sha: Git SHA of the PR head commit.
        base_sha: Git SHA of the PR base commit.
        decision: Gate decision outcome.
        tier: Evaluation tier that produced the decision.
        violations: List of rule violations detected.
        run_id: Optional pipeline run identifier.
        correlation_id: Optional correlation identifier for tracing.
        run_fingerprint: Optional fingerprint identifying the run configuration.
        decided_at: Timestamp when the gate decision was made.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    gate_id: UUID = Field(
        ..., description="Unique identifier for this gate evaluation."
    )
    pr_ref: str = Field(
        ...,
        description="Pull request reference (e.g. 'OmniNode-ai/omnibase_infra#42').",
    )
    head_sha: str = Field(..., description="Git SHA of the PR head commit.")
    base_sha: str = Field(..., description="Git SHA of the PR base commit.")
    decision: Literal["PASS", "WARN", "QUARANTINE"] = Field(
        ..., description="Gate decision outcome."
    )
    tier: Literal["tier-a", "tier-b"] = Field(
        ..., description="Evaluation tier that produced the decision."
    )
    violations: list[ModelMergeGateViolation] = Field(
        default_factory=list,
        description="List of rule violations detected.",
    )
    run_id: UUID | None = Field(
        default=None, description="Optional pipeline run identifier."
    )
    correlation_id: UUID | None = Field(
        default=None, description="Optional correlation identifier for tracing."
    )
    run_fingerprint: str | None = Field(
        default=None,
        description="Optional fingerprint identifying the run configuration.",
    )
    decided_at: datetime = Field(
        ..., description="Timestamp when the gate decision was made."
    )


__all__: list[str] = ["ModelMergeGateResult"]
