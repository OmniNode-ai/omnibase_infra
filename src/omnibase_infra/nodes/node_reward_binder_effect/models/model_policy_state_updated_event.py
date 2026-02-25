# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""PolicyStateUpdatedEvent emitted when policy state transitions.

Published to: ``onex.evt.omnimemory.policy-state-updated.v1``

Ticket: OMN-2552
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelPolicyStateUpdatedEvent(BaseModel):
    """Event emitted when policy state transitions.

    Published to: ``onex.evt.omnimemory.policy-state-updated.v1``

    Includes both ``old_state`` and ``new_state`` snapshots for auditability.
    Only emitted when the policy state actually changed between evaluations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier.",
    )
    run_id: UUID = Field(
        ...,
        description="Evaluation run ID that triggered the state transition.",
    )
    old_state: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot before the evaluation.",
    )
    new_state: dict[str, object] = Field(
        default_factory=dict,
        description="Policy state snapshot after the evaluation.",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event emission.",
    )


__all__: list[str] = ["ModelPolicyStateUpdatedEvent"]
