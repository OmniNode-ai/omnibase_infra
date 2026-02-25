# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Multi-dimensional score for a single evaluation target.

Stub model pending OMN-2537 merge (canonical models in omnibase_core).

Ticket: OMN-2552
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelScoreVector(BaseModel):
    """Multi-dimensional score for a single evaluation target.

    Stub pending OMN-2537 merge into omnibase_core.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: UUID = Field(..., description="ID of the target being scored.")
    target_type: Literal["tool", "model", "pattern", "agent"] = Field(
        ..., description="Category of the target."
    )
    dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="Named score dimensions (e.g. accuracy, latency_ms).",
    )
    composite_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Composite scalar in [-1, 1].",
    )


__all__: list[str] = ["ModelScoreVector"]
