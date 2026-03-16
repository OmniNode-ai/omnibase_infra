# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Domain event model for artifact update plans."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
    ModelImpactedArtifact,
)
from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_task import (
    ModelUpdateTask,
)


class ModelUpdatePlan(BaseModel):
    """A complete update plan produced by the REDUCER node.

    Aggregates impacted artifacts and their corresponding tasks,
    along with the overall merge policy for the source entity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: UUID
    source_trigger_id: UUID
    source_entity_ref: str
    summary: str
    impacted_artifacts: list[ModelImpactedArtifact]
    tasks: list[ModelUpdateTask]
    merge_policy: Literal["none", "warn", "require", "strict"]
    created_at: datetime
