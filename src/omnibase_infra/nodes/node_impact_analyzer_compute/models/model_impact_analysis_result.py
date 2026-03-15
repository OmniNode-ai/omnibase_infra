# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Domain event model for impact analysis results."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
    ModelImpactedArtifact,
)


class ModelImpactAnalysisResult(BaseModel):
    """Aggregated result of impact analysis for a single trigger.

    Contains the list of impacted artifacts and the highest merge
    policy derived from their update_policy values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_trigger_id: UUID
    impacted_artifacts: list[ModelImpactedArtifact]
    highest_merge_policy: Literal["none", "warn", "require", "strict"]
