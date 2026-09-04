# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-sensitive model-review runner-overlay observations."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelModelReviewCapabilityObservation(BaseModel):
    """Non-sensitive facts supplied by a runner-overlay health collector."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    runner_labels: frozenset[str] = Field(default_factory=frozenset)
    present_reference_ids: frozenset[UUID] = Field(default_factory=frozenset)
    healthy_reference_ids: frozenset[UUID] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def validate_health_assertions_are_present(
        self,
    ) -> ModelModelReviewCapabilityObservation:
        """Reject an impossible claim that a missing reference is healthy."""
        if not self.healthy_reference_ids <= self.present_reference_ids:
            msg = "healthy_reference_ids must be a subset of present_reference_ids"
            raise ValueError(msg)
        return self


__all__ = ["ModelModelReviewCapabilityObservation"]
