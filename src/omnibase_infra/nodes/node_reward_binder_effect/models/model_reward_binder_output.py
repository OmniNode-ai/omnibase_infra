# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Output model for the RewardBinder EFFECT node.

Ticket: OMN-2552
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRewardBinderOutput(BaseModel):
    """Result of a RewardBinderEffect operation.

    Reports which events were emitted and the computed objective fingerprint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether all events were emitted successfully.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )
    run_id: UUID = Field(
        ...,
        description="Evaluation run ID from the input EvaluationResult.",
    )
    objective_fingerprint: str = Field(
        ...,
        description="SHA-256 hex digest of the ObjectiveSpec (64 hex chars).",
    )
    reward_assigned_event_ids: tuple[UUID, ...] = Field(
        default_factory=tuple,
        description="Event IDs of all RewardAssignedEvents emitted (one per target).",
    )
    policy_state_updated_event_id: UUID | None = Field(
        default=None,
        description="Event ID of the PolicyStateUpdatedEvent (None if state unchanged).",
    )
    topics_published: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Kafka topic names that received events.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the operation failed.",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.success


__all__: list[str] = ["ModelRewardBinderOutput"]
