# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical gateway envelope model."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelGatewayEnvelope(BaseModel):
    """Tenant-bound envelope used at the gateway transform boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: UUID = Field(..., description="Immutable tenant identity.")
    tenant_slug: str = Field(..., description="Recorded human-facing tenant slug.")
    envelope_id: UUID = Field(..., description="Unique envelope id used for dedupe.")
    correlation_id: UUID = Field(..., description="Per-tenant request correlation id.")
    causation_id: str | None = Field(default=None, description="Parent envelope id.")
    event_type: str = Field(..., description="Raw routing key preserved across hops.")
    source_topic: str = Field(..., description="Topic this envelope was consumed from.")
    wire_topic: str = Field(..., description="Tenant-prefixed cloud topic.")
    canonical_topic: str = Field(..., description="Bare contract-declared topic.")
    payload: dict[str, object] = Field(default_factory=dict)

    @field_validator(
        "tenant_slug",
        "event_type",
        "source_topic",
        "canonical_topic",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("gateway envelope field must not be empty")
        return value.strip()
