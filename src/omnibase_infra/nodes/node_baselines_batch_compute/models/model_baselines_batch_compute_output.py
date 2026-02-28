# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Output model for NodeBaselinesBatchCompute.

Ticket: OMN-3043
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.services.observability.baselines.models.model_batch_compute_baselines_result import (
    ModelBatchComputeBaselinesResult,
)


class ModelBaselinesBatchComputeOutput(BaseModel):
    """Output of a baselines batch computation run.

    Attributes:
        result: Per-table row counts and any phase errors.
            ``result.total_rows`` gives total rows written.
            ``result.errors`` contains non-fatal phase failures.
        snapshot_emitted: True if the baselines-computed snapshot event
            was successfully published to Kafka; False otherwise
            (no publisher, publish failure, or zero rows written).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    result: ModelBatchComputeBaselinesResult
    snapshot_emitted: bool


__all__: list[str] = ["ModelBaselinesBatchComputeOutput"]
