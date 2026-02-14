# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""A/B Baseline Comparison Models.

Provides models for A/B baseline comparison infrastructure used to prove
pattern ROI during Tier 2+ promotion decisions.

Models:
    ModelCostMetrics: Token, time, and retry metrics for a single run.
    ModelOutcomeMetrics: Pass/fail, flake, and review metrics for a run.
    ModelBaselineRunConfig: Configuration for an A/B comparison run.
    ModelBaselineRunResult: Result of a single run variant (baseline/candidate).
    ModelCostDelta: Cost delta between baseline and candidate runs.
    ModelOutcomeDelta: Outcome delta between baseline and candidate runs.
    ModelAttributionRecord: Complete attribution record with deltas and ROI.

Tracking:
    - OMN-2155: Baselines + ROI -- A/B Run Infrastructure + Cost/Outcome Metrics
"""

from omnibase_infra.models.baseline.model_attribution_record import (
    ModelAttributionRecord,
)
from omnibase_infra.models.baseline.model_baseline_run_config import (
    ModelBaselineRunConfig,
)
from omnibase_infra.models.baseline.model_baseline_run_result import (
    ModelBaselineRunResult,
)
from omnibase_infra.models.baseline.model_cost_delta import ModelCostDelta
from omnibase_infra.models.baseline.model_cost_metrics import ModelCostMetrics
from omnibase_infra.models.baseline.model_outcome_delta import ModelOutcomeDelta
from omnibase_infra.models.baseline.model_outcome_metrics import ModelOutcomeMetrics

__all__: list[str] = [
    "ModelAttributionRecord",
    "ModelBaselineRunConfig",
    "ModelBaselineRunResult",
    "ModelCostDelta",
    "ModelCostMetrics",
    "ModelOutcomeDelta",
    "ModelOutcomeMetrics",
]
