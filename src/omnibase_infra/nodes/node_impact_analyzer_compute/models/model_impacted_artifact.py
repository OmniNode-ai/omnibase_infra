# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Domain event model for impacted artifacts."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelImpactedArtifact(BaseModel):
    """Represents a single artifact impacted by a change trigger.

    Produced by the impact analyzer COMPUTE node after matching
    changed files against the artifact registry's source triggers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: UUID
    artifact_type: Literal[
        "doc",
        "design_spec",
        "runbook",
        "roadmap",
        "reference",
        "migration_note",
        "release_note",
    ]
    path: str
    impact_strength: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str]
    required_action: Literal["none", "review", "patch", "regenerate", "create"]
