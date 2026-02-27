# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Local resolution event model for the event ledger publisher.

This model mirrors the essential fields of ``ModelResolutionEvent`` from
omnibase_core (created in PR #575). It exists here as a local schema so
the infra publisher can operate independently of the core package release
cycle.

TODO(OMN-2895): Replace this local model with the canonical import once
    omnibase_core PR #575 merges and a release containing
    ``omnibase_core.models.routing.model_resolution_event.ModelResolutionEvent``
    is available as a dependency.

Fields mirror the plan specification from Phase 6:
    - event_id: Unique identifier for this resolution event
    - timestamp: When the resolution was performed
    - registry_snapshot_hash: BLAKE3 hash of the provider registry at resolution time
    - policy_bundle_hash: SHA-256 of the policy bundle used
    - trust_graph_hash: SHA-256 of the trust graph used
    - tier_progression: Ordered list of tier attempts with timing
    - proofs_attempted: List of resolution proofs attempted
    - success: Whether resolution succeeded
    - failure_code: Structured failure code (if failed)
    - failure_reason: Human-readable failure description (if failed)
    - route_plan_id: UUID of the resulting route plan (if succeeded)
    - dependency_capability: The capability that was resolved

Related:
    - OMN-2895: Resolution Event Ledger (Phase 6 of OMN-2897 epic)
    - ModelResolutionEvent in omnibase_core (PR #575)
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.resolution.model_resolution_proof_local import (
    ModelResolutionProofLocal,
)
from omnibase_infra.services.resolution.model_tier_attempt_local import (
    ModelTierAttemptLocal,
)


class ModelResolutionEventLocal(BaseModel):
    """Local resolution event model for the event ledger.

    This model represents a resolution decision audit event, recording the
    full context and outcome of a tiered dependency resolution attempt.

    TODO(OMN-2895): Replace with ``ModelResolutionEvent`` from omnibase_core
        once PR #575 merges and is available as a dependency.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this event"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the resolution was performed",
    )
    dependency_capability: str = Field(..., description="The capability being resolved")
    registry_snapshot_hash: str = Field(
        default="", description="BLAKE3 hash of the provider registry"
    )
    policy_bundle_hash: str = Field(
        default="", description="SHA-256 of the policy bundle used"
    )
    trust_graph_hash: str = Field(
        default="", description="SHA-256 of the trust graph used"
    )
    route_plan_id: UUID | None = Field(
        default=None,
        description="UUID of the resulting route plan (if resolution succeeded)",
    )
    tier_progression: tuple[ModelTierAttemptLocal, ...] = Field(
        default_factory=tuple,
        description="Ordered list of tier attempts with timing",
    )
    proofs_attempted: tuple[ModelResolutionProofLocal, ...] = Field(
        default_factory=tuple,
        description="List of resolution proofs attempted",
    )
    success: bool = Field(..., description="Whether the resolution succeeded")
    failure_code: str | None = Field(
        default=None, description="Structured failure code"
    )
    failure_reason: str | None = Field(
        default=None, description="Human-readable failure description"
    )

    def to_publishable_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict suitable for event bus publishing.

        Returns:
            Dictionary with all fields serialized to JSON-safe types.
            UUIDs are serialized as strings, datetimes as ISO 8601 strings,
            nested models are recursively serialized.
        """
        return self.model_dump(mode="json")


__all__: list[str] = ["ModelResolutionEventLocal"]
