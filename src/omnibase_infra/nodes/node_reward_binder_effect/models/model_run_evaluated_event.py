# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""RunEvaluatedEvent emitted after every evaluation run.

Published to: ``onex.evt.omnimemory.run-evaluated.v1``

Ticket: OMN-2927
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

    Uses canonical omnibase_core.ModelScoreVector field shapes inline
    (correctness, safety, cost, latency, maintainability, human_time).
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
    correctness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Gate-derived correctness from canonical ModelScoreVector.",
    )
    safety: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Security, PII, and blacklist gate composite score.",
    )
    cost: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Inverted cost score: lower cost = higher score.",
    )
    latency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Inverted latency score: lower latency = higher score.",
    )
    maintainability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cyclomatic complexity delta and test coverage composite score.",
    )
    human_time: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Inverted human intervention score: fewer retries/reviews = higher score.",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event emission.",
    )


__all__: list[str] = ["ModelRunEvaluatedEvent"]
