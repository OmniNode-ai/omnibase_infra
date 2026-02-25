# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""RunEvaluatedEvent emitted after every evaluation run.

Published to: ``onex.evt.omnimemory.run-evaluated.v1``

Ticket: OMN-2552
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelRunEvaluatedEvent(BaseModel):
    """Event emitted after every evaluation run.

    Published to: ``onex.evt.omnimemory.run-evaluated.v1``

    The ``objective_fingerprint`` is a SHA-256 hex digest of the serialized
    ``ModelObjectiveSpec`` used for the run — providing tamper-evident traceability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier.",
    )
    run_id: UUID = Field(
        ...,
        description="Evaluation run ID from ModelEvaluationResult.",
    )
    objective_id: UUID = Field(
        ...,
        description="Objective that drove this evaluation.",
    )
    objective_fingerprint: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of ModelObjectiveSpec.model_dump_json() — tamper-evident.",
    )
    composite_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of target_id (str) -> composite_score for quick access.",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event emission.",
    )


__all__: list[str] = ["ModelRunEvaluatedEvent"]
