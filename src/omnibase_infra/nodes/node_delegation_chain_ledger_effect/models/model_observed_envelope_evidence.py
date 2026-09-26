# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure, recorded envelope evidence for delegation graph replay (OMN-19729)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelObservedEnvelopeEvidence(BaseModel):
    """One envelope after its boundary adapter has extracted durable evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    envelope_id: UUID
    topic: str = Field(min_length=1)
    parent_envelope_id: UUID | None = None
    correlation_id: UUID
    evidence_fingerprint: str = Field(min_length=1)
    observed_index: int = Field(ge=0)
    partition: int = Field(ge=0)
    kafka_offset: int = Field(ge=0)

    @property
    def identity_claim(self) -> tuple[str, UUID | None, UUID, str]:
        """Fields that must agree for two deliveries to be one envelope."""
        return (
            self.topic,
            self.parent_envelope_id,
            self.correlation_id,
            self.evidence_fingerprint,
        )


__all__ = ["ModelObservedEnvelopeEvidence"]
