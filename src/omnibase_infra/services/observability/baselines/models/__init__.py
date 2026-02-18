# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Models for the baselines observability service.

Provides row models for the three baselines tables and the batch
computation result model.

Models:
    ModelBaselinesComparisonRow: Row from baselines_comparisons table.
    ModelBaselinesTrendRow: Row from baselines_trend table.
    ModelBaselinesBreakdownRow: Row from baselines_breakdown table.
    ModelBatchComputeBaselinesResult: Result of a batch computation run.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from omnibase_infra.services.observability.baselines.models.model_baselines_breakdown_row import (
    ModelBaselinesBreakdownRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_comparison_row import (
    ModelBaselinesComparisonRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_trend_row import (
    ModelBaselinesTrendRow,
)
from omnibase_infra.services.observability.baselines.models.model_batch_compute_baselines_result import (
    ModelBatchComputeBaselinesResult,
)

__all__: list[str] = [
    "ModelBaselinesBreakdownRow",
    "ModelBaselinesComparisonRow",
    "ModelBaselinesTrendRow",
    "ModelBatchComputeBaselinesResult",
]
