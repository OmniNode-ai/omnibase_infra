# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Savings estimation input model.

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_savings_estimation_compute.models.model_effectiveness_entry import (
    ModelEffectivenessEntry,
)


class ModelSavingsEstimationInput(BaseModel):
    """Input for savings estimation computation.

    Aggregates effectiveness data from one or more sessions for a
    single savings estimation event.

    Note: session_id and actual_model_id use ``str`` intentionally because
    they are free-form identifiers (not UUIDs) that must pass through to
    the Kafka payload as-is for omnidash projection compatibility.

    Attributes:
        session_id: Session identifier for this estimation batch.
        correlation_id: Correlation ID for tracing.
        effectiveness_entries: List of effectiveness measurements.
        actual_total_tokens: Total tokens consumed in the session.
        actual_model_id: Actual model used (e.g. 'claude-opus-4-6').
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    session_id: str = Field(  # noqa: onex-pattern-validation
        ..., min_length=1, description="Session identifier"
    )
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for tracing"
    )
    effectiveness_entries: tuple[ModelEffectivenessEntry, ...] = Field(
        ..., min_length=1, description="Effectiveness measurements"
    )
    actual_total_tokens: int = Field(
        default=0, ge=0, description="Total tokens consumed in session"
    )
    actual_model_id: str = Field(  # noqa: onex-pattern-validation
        default="claude-opus-4-6", description="Actual model identifier"
    )


__all__: list[str] = ["ModelSavingsEstimationInput"]
