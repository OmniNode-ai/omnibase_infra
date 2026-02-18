# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Snapshot event payload for the baselines-computed Kafka event.

This model is the stable contract emitted to
``onex.evt.omnibase-infra.baselines-computed.v1`` after each successful
``ServiceBatchComputeBaselines.compute_and_persist()`` run.

The omnidash consumer projects this snapshot into the local
``omnidash_analytics`` read-model, replacing the previous snapshot.

Contract stability:
    - ``snapshot_id`` is a UUID generated fresh at each emit call.
    - ``contract_version`` is incremented only on breaking schema changes.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
    - OMN-2331: Wire /api/baselines/* to real tables -- remove mockOnEmpty
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.observability.baselines.models.model_baselines_breakdown_row import (
    ModelBaselinesBreakdownRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_comparison_row import (
    ModelBaselinesComparisonRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_trend_row import (
    ModelBaselinesTrendRow,
)


class ModelBaselinesSnapshotEvent(BaseModel):
    """Full baselines snapshot emitted after each batch computation.

    Consumers should replace their local projection on receipt.

    Attributes:
        snapshot_id: UUID generated fresh at emit time for this specific snapshot.
        contract_version: Schema version (bump only on breaking changes).
        computed_at_utc: When the batch computation completed.
        window_start_utc: Earliest data point in the computation window.
        window_end_utc: Latest data point in the computation window.
        comparisons: Daily treatment vs control comparison rows.
        trend: Per-cohort per-day trend rows.
        breakdown: Per-pattern treatment vs control breakdown rows.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: UUID = Field(
        description="UUID identifying this specific computation run.",
    )
    contract_version: int = Field(
        default=1,
        description="Schema version; bump only on breaking changes.",
    )
    computed_at_utc: datetime = Field(
        description="When the batch computation completed.",
    )
    window_start_utc: datetime | None = Field(
        default=None,
        description="Earliest data point in the computation window.",
    )
    window_end_utc: datetime | None = Field(
        default=None,
        description="Latest data point in the computation window.",
    )
    comparisons: list[ModelBaselinesComparisonRow] = Field(
        default_factory=list,
        description="Daily treatment vs control comparison rows.",
    )
    trend: list[ModelBaselinesTrendRow] = Field(
        default_factory=list,
        description="Per-cohort per-day trend rows.",
    )
    breakdown: list[ModelBaselinesBreakdownRow] = Field(
        default_factory=list,
        description="Per-pattern treatment vs control breakdown rows.",
    )


__all__: list[str] = ["ModelBaselinesSnapshotEvent"]
