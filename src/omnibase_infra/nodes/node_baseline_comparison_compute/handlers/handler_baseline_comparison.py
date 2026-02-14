# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for computing A/B baseline comparison deltas and ROI.

Takes paired baseline and candidate run results, computes cost and
outcome deltas, and produces a :class:`ModelAttributionRecord` that
proves (or disproves) pattern ROI for promotion decisions.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.models.baseline.model_attribution_record import (
    ModelAttributionRecord,
)
from omnibase_infra.models.baseline.model_cost_delta import ModelCostDelta
from omnibase_infra.models.baseline.model_outcome_delta import ModelOutcomeDelta
from omnibase_infra.nodes.node_baseline_comparison_compute.models import (
    ModelBaselineComparisonInput,
)

logger = logging.getLogger(__name__)


class HandlerBaselineComparison:
    """Compute cost/outcome deltas and ROI from paired A/B run results.

    This handler is stateless and pure.  It receives the paired
    baseline and candidate run results, computes deltas using the
    ``from_metrics`` factory methods on ``ModelCostDelta`` and
    ``ModelOutcomeDelta``, and returns a ``ModelAttributionRecord``.

    ROI is considered positive when:
    - Token savings are non-negative (candidate uses same or fewer tokens), AND
    - Quality is maintained or improved (outcome delta shows no regression).

    Note:
        This is an infrastructure handler (``INFRA_HANDLER``) with
        ``NONDETERMINISTIC_COMPUTE`` category because it performs
        computation with no external I/O but uses ``uuid4()`` and
        ``datetime.now()``, making it non-deterministic.
    """

    # ------------------------------------------------------------------
    # Handler classification
    # ------------------------------------------------------------------

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-baseline-comparison"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for comparison.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            handler that computes baseline comparison deltas.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: non-deterministic computation.

        Returns:
            EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE - This handler
            performs computation with no external I/O but uses uuid4() and
            datetime.now(), making it non-deterministic.
        """
        return EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE

    # ------------------------------------------------------------------
    # Core handle method
    # ------------------------------------------------------------------

    async def handle(
        self,
        comparison_input: ModelBaselineComparisonInput,
    ) -> ModelAttributionRecord:
        """Compute A/B comparison deltas and produce an attribution record.

        Args:
            comparison_input: Paired baseline and candidate run results
                with the original comparison configuration.

        Returns:
            A ``ModelAttributionRecord`` containing cost deltas, outcome
            deltas, and an ROI determination.
        """
        config = comparison_input.config
        baseline = comparison_input.baseline_result
        candidate = comparison_input.candidate_result

        # Fallback to auto-generated correlation_id per repo guidelines
        correlation_id = (
            config.correlation_id if config.correlation_id is not None else uuid4()
        )

        # Compute deltas
        cost_delta = ModelCostDelta.from_metrics(
            baseline=baseline.cost_metrics,
            candidate=candidate.cost_metrics,
        )
        outcome_delta = ModelOutcomeDelta.from_metrics(
            baseline=baseline.outcome_metrics,
            candidate=candidate.outcome_metrics,
        )

        # ROI is positive when:
        # 1. Token savings are non-negative (no cost regression)
        # 2. Quality is maintained or improved
        roi_positive = self._compute_roi(cost_delta, outcome_delta)

        record_id = uuid4()

        logger.info(
            "Baseline comparison completed: pattern=%s, "
            "token_savings=%.1f%%, time_savings=%.1f%%, "
            "quality_improved=%s, roi_positive=%s (cid=%s)",
            config.pattern_id,
            cost_delta.token_savings_pct,
            cost_delta.time_savings_pct,
            outcome_delta.quality_improved,
            roi_positive,
            correlation_id,
        )

        return ModelAttributionRecord(
            record_id=record_id,
            pattern_id=config.pattern_id,
            scenario_id=config.scenario_id,
            correlation_id=correlation_id,
            current_tier=config.current_tier,
            target_tier=config.target_tier,
            baseline_result=baseline,
            candidate_result=candidate,
            cost_delta=cost_delta,
            outcome_delta=outcome_delta,
            roi_positive=roi_positive,
            created_at=datetime.now(tz=UTC),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_roi(
        cost_delta: ModelCostDelta,
        outcome_delta: ModelOutcomeDelta,
    ) -> bool:
        """Determine whether the pattern demonstrates positive ROI.

        ROI is positive when:
        - Token delta is non-negative (candidate uses same or fewer tokens)
        - Candidate passed validation
        - Quality did not regress (outcome is same or better)

        Args:
            cost_delta: Cost delta between baseline and candidate.
            outcome_delta: Outcome delta between baseline and candidate.

        Returns:
            True if the pattern demonstrates positive ROI.
        """
        # Cost must not increase (non-negative delta = savings or neutral)
        cost_acceptable = cost_delta.token_delta >= 0

        # Candidate must pass
        candidate_passed = outcome_delta.candidate_passed

        # Quality must not regress
        no_quality_regression = (
            outcome_delta.quality_improved or outcome_delta.check_delta >= 0
        )

        return cost_acceptable and candidate_passed and no_quality_regression


__all__: list[str] = ["HandlerBaselineComparison"]
