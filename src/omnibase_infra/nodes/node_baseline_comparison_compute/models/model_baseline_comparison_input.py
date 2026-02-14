# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Input model for the baseline comparison compute node.

Pairs the baseline and candidate run results with the original
configuration for delta computation.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums import EnumRunVariant
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

    @model_validator(mode="after")
    def _validate_variant_pairing(self) -> Self:
        """Validate that baseline and candidate results have correct variants.

        Prevents silently inverted deltas from swapped run results.

        Raises:
            ValueError: If baseline_result.variant is not BASELINE or
                candidate_result.variant is not CANDIDATE.
        """
        if self.baseline_result.variant != EnumRunVariant.BASELINE:
            msg = (
                f"baseline_result.variant must be BASELINE, "
                f"got {self.baseline_result.variant}"
            )
            raise ValueError(msg)
        if self.candidate_result.variant != EnumRunVariant.CANDIDATE:
            msg = (
                f"candidate_result.variant must be CANDIDATE, "
                f"got {self.candidate_result.variant}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_correlation_id_match(self) -> Self:
        """Validate that baseline and candidate results share the same correlation_id.

        Prevents comparing results from different A/B comparison runs.
        Two results with different correlation_ids originate from independent
        run executions and must not be combined into a single comparison.

        Raises:
            ValueError: If baseline_result.correlation_id differs from
                candidate_result.correlation_id.
        """
        if self.baseline_result.correlation_id != self.candidate_result.correlation_id:
            msg = (
                f"baseline_result.correlation_id "
                f"({self.baseline_result.correlation_id}) must match "
                f"candidate_result.correlation_id "
                f"({self.candidate_result.correlation_id}); "
                f"results from different comparison runs cannot be paired"
            )
            raise ValueError(msg)
        return self


__all__: list[str] = ["ModelBaselineComparisonInput"]
