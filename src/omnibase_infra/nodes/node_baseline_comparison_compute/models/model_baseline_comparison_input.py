# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for the baseline comparison compute node.

Pairs the baseline and candidate run results with the original
configuration for delta computation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.baseline.model_baseline_run_config import (
    ModelBaselineRunConfig,
)
from omnibase_infra.models.baseline.model_baseline_run_result import (
    ModelBaselineRunResult,
)


class ModelBaselineComparisonInput(BaseModel):
    """Input for the baseline comparison compute node.

    Attributes:
        config: Configuration for the A/B comparison run.
        baseline_result: Result of the baseline (no pattern) run.
        candidate_result: Result of the candidate (with pattern) run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    config: ModelBaselineRunConfig = Field(
        ...,
        description="Configuration for the A/B comparison run.",
    )
    baseline_result: ModelBaselineRunResult = Field(
        ...,
        description="Result of the baseline (no pattern) run.",
    )
    candidate_result: ModelBaselineRunResult = Field(
        ...,
        description="Result of the candidate (with pattern) run.",
    )


__all__: list[str] = ["ModelBaselineComparisonInput"]
