# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Specification of an objective used to drive evaluation.

Stub model pending OMN-2537 merge (canonical models in omnibase_core).

Ticket: OMN-2552
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelObjectiveSpec(BaseModel):
    """Specification of an objective used to drive evaluation.

    Stub pending OMN-2537 merge into omnibase_core.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_id: UUID = Field(
        default_factory=uuid4, description="Unique objective ID."
    )
    name: str = Field(..., description="Human-readable objective name.")
    description: str = Field(default="", description="Objective description.")
    target_types: tuple[Literal["tool", "model", "pattern", "agent"], ...] = Field(
        default_factory=tuple,
        description="Target types this objective applies to.",
    )
    weight: float = Field(default=1.0, ge=0.0, description="Relative weight.")


__all__: list[str] = ["ModelObjectiveSpec"]
