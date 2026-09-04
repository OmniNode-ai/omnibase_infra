# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-sensitive model-review runner-overlay observations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_REVIEW_OBSERVATION_PROVENANCE = "runner-local-model-review-preflight"
MODEL_REVIEW_ATTESTATION_NAMESPACE = UUID("8abf4e58-80ba-5f76-9d38-8787b4e47b8e")


class ModelModelReviewCapabilityObservation(BaseModel):
    """Non-sensitive facts supplied by a runner-overlay health collector."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    runner_labels: frozenset[str] = Field(default_factory=frozenset)
    runner_groups: frozenset[str] = Field(default_factory=frozenset)
    present_reference_ids: frozenset[UUID] = Field(default_factory=frozenset)
    healthy_reference_ids: frozenset[UUID] = Field(default_factory=frozenset)
    reviewer_cli_available: bool = False
    observed_at: datetime | None = None
    attestation_id: UUID | None = None
    provenance: Literal["runner-local-model-review-preflight"] | None = None

    @model_validator(mode="after")
    def validate_health_assertions_are_present(
        self,
    ) -> ModelModelReviewCapabilityObservation:
        """Reject an impossible claim that a missing reference is healthy."""
        if not self.healthy_reference_ids <= self.present_reference_ids:
            msg = "healthy_reference_ids must be a subset of present_reference_ids"
            raise ValueError(msg)
        return self

    def derived_attestation_id(self) -> UUID:
        """Derive the attestation identity from the complete probe output.

        The runner-local collector is the only production entry point that
        creates observations. Preflight recomputes this value so a detached or
        caller-supplied UUID cannot be presented as probe evidence.
        """
        payload = {
            "runner_groups": sorted(self.runner_groups),
            "runner_labels": sorted(self.runner_labels),
            "present_reference_ids": sorted(
                str(reference_id) for reference_id in self.present_reference_ids
            ),
            "healthy_reference_ids": sorted(
                str(reference_id) for reference_id in self.healthy_reference_ids
            ),
            "reviewer_cli_available": self.reviewer_cli_available,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
            "provenance": self.provenance,
        }
        return uuid5(
            MODEL_REVIEW_ATTESTATION_NAMESPACE,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )


__all__ = [
    "MODEL_REVIEW_ATTESTATION_NAMESPACE",
    "MODEL_REVIEW_OBSERVATION_PROVENANCE",
    "ModelModelReviewCapabilityObservation",
]
