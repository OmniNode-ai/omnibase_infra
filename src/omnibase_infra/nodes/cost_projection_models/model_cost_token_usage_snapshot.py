# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost token-usage projection snapshot model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_infra.nodes.cost_projection_models.model_cost_token_usage_snapshot_row import (
    ModelCostTokenUsageSnapshotRow,
)
from omnibase_infra.services.cost_api.model_types import AggregationWindow


class ModelCostTokenUsageSnapshot(BaseModel):
    """Frozen payload for cost token-usage projection snapshots."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    window: AggregationWindow
    rows: list[ModelCostTokenUsageSnapshotRow]
    snapshot_timestamp: datetime

    @field_validator("snapshot_timestamp")
    @classmethod
    def validate_snapshot_tz_aware(cls, value: datetime) -> datetime:
        """Require timezone-aware snapshot timestamps."""
        if value.tzinfo is None:
            raise ValueError("snapshot_timestamp must be timezone-aware")
        return value


__all__ = ["ModelCostTokenUsageSnapshot"]
