# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed event published when a runtime error incident is escalated (OMN-21029)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.enum_escalation_reason import (
    EnumEscalationReason,
)


class ModelRuntimeErrorEscalatedEvent(BaseModel):
    """One incident crossed the escalation rule. Published once per incident."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID = Field(
        ..., description="Correlation ID of the runtime error event that crossed."
    )
    fingerprint: str = Field(..., description="Incident fingerprint.")
    logger_family: str = Field(..., description="Logger that emitted the error.")
    error_category: EnumRuntimeErrorCategory
    severity: EnumRuntimeErrorSeverity
    message_template: str = ""
    occurrence_count: int = Field(..., ge=1)
    reason: EnumEscalationReason
    matched_rule: str = Field(..., description="Triage rule that matched the error.")
    service_label: str = ""
    hostname: str = ""
    correlated_consumer_fingerprint: str = ""
    escalated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def partition_key(self) -> str:
        """Keep every escalation of one fingerprint on the same partition."""
        return self.fingerprint


__all__ = ["ModelRuntimeErrorEscalatedEvent"]
