# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Payload model for the write-bundle intent.

This payload drives HandlerWriteBundle which performs an idempotent
INSERT into delta_bundles ON CONFLICT (pr_ref, head_sha) DO NOTHING.

Related Tickets:
    - OMN-3142: NodeDeltaBundleEffect implementation
    - Migration 039: delta_bundles table
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadWriteBundle(BaseModel):
    """Payload for the delta-bundle write intent.

    Carries all fields required for the idempotent INSERT into delta_bundles.
    Fix-PR detection is performed by the handler based on the labels field.

    Attributes:
        intent_type: Routing key -- always "delta_bundle.write_bundle".
        correlation_id: Correlation UUID from the originating context.
        bundle_id: Unique event identifier from the merge-gate-decision event.
        pr_ref: PR reference string (e.g. "owner/repo#123").
        head_sha: Git HEAD SHA at time of gate decision.
        base_sha: Git base SHA for the PR diff.
        coding_model: LLM model that authored the code.
        subsystem: Subsystem classification.
        gate_decision: Merge-gate verdict (PASS, WARN, QUARANTINE).
        gate_violations: JSON-serialisable list of gate violation details.
        labels: List of PR labels. Handler parses for stabilizes:<pr_ref>.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["delta_bundle.write_bundle"] = Field(
        default="delta_bundle.write_bundle",
        description="Routing key for this intent.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation UUID from the originating pipeline/session.",
    )
    bundle_id: UUID = Field(
        ...,
        description="Unique event identifier from the merge-gate-decision event.",
    )
    pr_ref: str = Field(
        ...,
        description="PR reference string (e.g. 'owner/repo#123').",
    )
    head_sha: str = Field(
        ...,
        description="Git HEAD SHA at time of gate decision.",
    )
    base_sha: str = Field(
        ...,
        description="Git base SHA for the PR diff.",
    )
    coding_model: str | None = Field(
        default=None,
        description="LLM model that authored the code.",
    )
    subsystem: str | None = Field(
        default=None,
        description="Subsystem classification (e.g. 'omnibase_infra').",
    )
    gate_decision: Literal["PASS", "WARN", "QUARANTINE"] = Field(
        ...,
        description="Merge-gate verdict.",
    )
    gate_violations: list[dict[str, object]] = Field(
        default_factory=list,
        description="JSON-serialisable list of gate violation details.",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="PR labels. Handler parses for 'stabilizes:<pr_ref>'.",
    )


__all__: list[str] = ["ModelPayloadWriteBundle"]
