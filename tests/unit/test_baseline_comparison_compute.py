# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for the A/B baseline comparison compute node.

Tests:
- ModelCostMetrics construction and defaults
- ModelOutcomeMetrics construction and validation
- ModelBaselineRunConfig.requires_baseline() tier logic
- ModelCostDelta.from_metrics() delta computation
- ModelOutcomeDelta.from_metrics() delta and quality improvement logic
- ModelBaselineRunResult __bool__ behaviour
- ModelAttributionRecord construction
- HandlerBaselineComparison handler properties and async handle()
- HandlerBaselineComparison._compute_roi() logic
- EnumRunVariant enum values
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLifecycleTier,
    EnumRunVariant,
)
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
from omnibase_infra.nodes.node_baseline_comparison_compute.handlers.handler_baseline_comparison import (
    HandlerBaselineComparison,
)
from omnibase_infra.nodes.node_baseline_comparison_compute.models import (
    ModelBaselineComparisonInput,
)

pytestmark = pytest.mark.unit

# ============================================================================
# Helpers
# ============================================================================


def _make_cost(
    total_tokens: int = 1000,
    prompt_tokens: int = 600,
    completion_tokens: int = 400,
    wall_time_ms: float = 5000.0,
    retry_count: int = 1,
) -> ModelCostMetrics:
    """Create cost metrics for testing."""
    return ModelCostMetrics(
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_time_ms=wall_time_ms,
        retry_count=retry_count,
    )


def _make_outcome(
    passed: bool = True,
    total_checks: int = 10,
    passed_checks: int = 9,
    failed_checks: int = 1,
    flake_rate: float = 0.05,
    review_iterations: int = 2,
) -> ModelOutcomeMetrics:
    """Create outcome metrics for testing."""
    return ModelOutcomeMetrics(
        passed=passed,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        flake_rate=flake_rate,
        review_iterations=review_iterations,
    )


def _make_run_result(
    variant: EnumRunVariant = EnumRunVariant.BASELINE,
    cost: ModelCostMetrics | None = None,
    outcome: ModelOutcomeMetrics | None = None,
    error: str = "",
    correlation_id: UUID | None = None,
) -> ModelBaselineRunResult:
    """Create a run result for testing."""
    now = datetime.now(tz=UTC)
    return ModelBaselineRunResult(
        run_id=uuid4(),
        variant=variant,
        correlation_id=correlation_id or uuid4(),
        cost_metrics=cost or _make_cost(),
        outcome_metrics=outcome or _make_outcome(),
        started_at=now,
        completed_at=now,
        error=error,
    )


def _make_config(
    current_tier: EnumLifecycleTier = EnumLifecycleTier.SUGGESTED,
    target_tier: EnumLifecycleTier = EnumLifecycleTier.SHADOW_APPLY,
) -> ModelBaselineRunConfig:
    """Create a baseline run config for testing."""
    return ModelBaselineRunConfig(
        pattern_id=uuid4(),
        scenario_id=uuid4(),
        correlation_id=uuid4(),
        current_tier=current_tier,
        target_tier=target_tier,
    )


def _make_comparison_input(
    config: ModelBaselineRunConfig | None = None,
    baseline: ModelBaselineRunResult | None = None,
    candidate: ModelBaselineRunResult | None = None,
) -> ModelBaselineComparisonInput:
    """Create a comparison input for testing.

    When baseline or candidate results are not provided, auto-generated
    results share the same ``correlation_id`` so the correlation-match
    validator passes.  Explicitly supplied results are used as-is.
    """
    shared_correlation_id = uuid4()
    return ModelBaselineComparisonInput(
        config=config or _make_config(),
        baseline_result=baseline
        or _make_run_result(
            variant=EnumRunVariant.BASELINE,
            correlation_id=shared_correlation_id,
        ),
        candidate_result=candidate
        or _make_run_result(
            variant=EnumRunVariant.CANDIDATE,
            correlation_id=shared_correlation_id,
        ),
    )


# ============================================================================
# EnumRunVariant
# ============================================================================


class TestEnumRunVariant:
    """Tests for EnumRunVariant enum."""

    def test_baseline_value(self) -> None:
        """BASELINE has value 'baseline'."""
        assert EnumRunVariant.BASELINE.value == "baseline"

    def test_candidate_value(self) -> None:
        """CANDIDATE has value 'candidate'."""
        assert EnumRunVariant.CANDIDATE.value == "candidate"

    def test_str_representation(self) -> None:
        """str() returns the value."""
        assert str(EnumRunVariant.BASELINE) == "baseline"
        assert str(EnumRunVariant.CANDIDATE) == "candidate"

    def test_from_string(self) -> None:
        """Enum can be constructed from string values."""
        assert EnumRunVariant("baseline") == EnumRunVariant.BASELINE
        assert EnumRunVariant("candidate") == EnumRunVariant.CANDIDATE


# ============================================================================
# ModelCostMetrics -- Construction
# ============================================================================


class TestCostMetricsConstruction:
    """Tests for ModelCostMetrics construction and defaults."""

    def test_default_values(self) -> None:
        """All fields default to zero."""
        metrics = ModelCostMetrics()
        assert metrics.total_tokens == 0
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.wall_time_ms == 0.0
        assert metrics.retry_count == 0

    def test_custom_values(self) -> None:
        """Custom values are accepted."""
        metrics = _make_cost()
        assert metrics.total_tokens == 1000
        assert metrics.prompt_tokens == 600
        assert metrics.completion_tokens == 400
        assert metrics.wall_time_ms == 5000.0
        assert metrics.retry_count == 1

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        metrics = _make_cost()
        with pytest.raises(ValidationError):
            metrics.total_tokens = 0  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelCostMetrics(
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_negative_tokens_rejected(self) -> None:
        """Negative token counts are rejected."""
        with pytest.raises(ValidationError):
            ModelCostMetrics(total_tokens=-1)

    def test_negative_wall_time_rejected(self) -> None:
        """Negative wall time is rejected."""
        with pytest.raises(ValidationError):
            ModelCostMetrics(wall_time_ms=-1.0)


# ============================================================================
# ModelOutcomeMetrics -- Construction
# ============================================================================


class TestOutcomeMetricsConstruction:
    """Tests for ModelOutcomeMetrics construction and defaults."""

    def test_required_passed_field(self) -> None:
        """passed is a required field."""
        with pytest.raises(ValidationError):
            ModelOutcomeMetrics()  # type: ignore[call-arg]

    def test_default_counters(self) -> None:
        """Optional counters default to zero."""
        metrics = ModelOutcomeMetrics(passed=True)
        assert metrics.total_checks == 0
        assert metrics.passed_checks == 0
        assert metrics.failed_checks == 0
        assert metrics.flake_rate == 0.0
        assert metrics.review_iterations == 0

    def test_flake_rate_range(self) -> None:
        """Flake rate must be in [0.0, 1.0]."""
        # Valid
        ModelOutcomeMetrics(passed=True, flake_rate=0.0)
        ModelOutcomeMetrics(passed=True, flake_rate=1.0)
        # Invalid
        with pytest.raises(ValidationError):
            ModelOutcomeMetrics(passed=True, flake_rate=1.5)
        with pytest.raises(ValidationError):
            ModelOutcomeMetrics(passed=True, flake_rate=-0.1)

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        metrics = _make_outcome()
        with pytest.raises(ValidationError):
            metrics.passed = False  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelOutcomeMetrics(
                passed=True,
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ============================================================================
# ModelBaselineRunConfig -- requires_baseline
# ============================================================================


class TestBaselineRunConfig:
    """Tests for ModelBaselineRunConfig construction and requires_baseline."""

    def test_construction(self) -> None:
        """Config can be constructed with all required fields."""
        config = _make_config()
        assert config.pattern_id is not None
        assert config.scenario_id is not None
        assert config.correlation_id is not None

    def test_requires_baseline_observed_to_suggested(self) -> None:
        """OBSERVED -> SUGGESTED does NOT require baseline (Tier 0->1)."""
        config = _make_config(
            current_tier=EnumLifecycleTier.OBSERVED,
            target_tier=EnumLifecycleTier.SUGGESTED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_suggested_to_shadow(self) -> None:
        """SUGGESTED -> SHADOW_APPLY DOES require baseline (Tier 1->2)."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.SHADOW_APPLY,
        )
        assert config.requires_baseline() is True

    def test_requires_baseline_shadow_to_promoted(self) -> None:
        """SHADOW_APPLY -> PROMOTED DOES require baseline (Tier 2->3)."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SHADOW_APPLY,
            target_tier=EnumLifecycleTier.PROMOTED,
        )
        assert config.requires_baseline() is True

    def test_requires_baseline_promoted_to_default(self) -> None:
        """PROMOTED -> DEFAULT DOES require baseline (Tier 3->4)."""
        config = _make_config(
            current_tier=EnumLifecycleTier.PROMOTED,
            target_tier=EnumLifecycleTier.DEFAULT,
        )
        assert config.requires_baseline() is True

    def test_requires_baseline_default(self) -> None:
        """DEFAULT tier does NOT require baseline (already at top)."""
        config = _make_config(
            current_tier=EnumLifecycleTier.DEFAULT,
            target_tier=EnumLifecycleTier.DEFAULT,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_suppressed(self) -> None:
        """SUPPRESSED tier does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SUPPRESSED,
            target_tier=EnumLifecycleTier.OBSERVED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_noop_suggested(self) -> None:
        """No-op SUGGESTED -> SUGGESTED does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.SUGGESTED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_noop_shadow_apply(self) -> None:
        """No-op SHADOW_APPLY -> SHADOW_APPLY does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SHADOW_APPLY,
            target_tier=EnumLifecycleTier.SHADOW_APPLY,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_noop_promoted(self) -> None:
        """No-op PROMOTED -> PROMOTED does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.PROMOTED,
            target_tier=EnumLifecycleTier.PROMOTED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_demotion_promoted_to_shadow(self) -> None:
        """Demotion PROMOTED -> SHADOW_APPLY does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.PROMOTED,
            target_tier=EnumLifecycleTier.SHADOW_APPLY,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_demotion_shadow_to_suggested(self) -> None:
        """Demotion SHADOW_APPLY -> SUGGESTED does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SHADOW_APPLY,
            target_tier=EnumLifecycleTier.SUGGESTED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_demotion_promoted_to_observed(self) -> None:
        """Demotion PROMOTED -> OBSERVED does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.PROMOTED,
            target_tier=EnumLifecycleTier.OBSERVED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_target_suppressed(self) -> None:
        """Transition to SUPPRESSED does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.SUPPRESSED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_default_to_observed(self) -> None:
        """Demotion DEFAULT -> OBSERVED does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.DEFAULT,
            target_tier=EnumLifecycleTier.OBSERVED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_default_to_promoted(self) -> None:
        """Demotion DEFAULT -> PROMOTED does NOT require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.DEFAULT,
            target_tier=EnumLifecycleTier.PROMOTED,
        )
        assert config.requires_baseline() is False

    def test_requires_baseline_skip_suggested_to_promoted(self) -> None:
        """Skip-promotion SUGGESTED -> PROMOTED DOES require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.PROMOTED,
        )
        assert config.requires_baseline() is True

    def test_requires_baseline_skip_suggested_to_default(self) -> None:
        """Skip-promotion SUGGESTED -> DEFAULT DOES require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.DEFAULT,
        )
        assert config.requires_baseline() is True

    def test_requires_baseline_skip_shadow_to_default(self) -> None:
        """Skip-promotion SHADOW_APPLY -> DEFAULT DOES require baseline."""
        config = _make_config(
            current_tier=EnumLifecycleTier.SHADOW_APPLY,
            target_tier=EnumLifecycleTier.DEFAULT,
        )
        assert config.requires_baseline() is True

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        config = _make_config()
        with pytest.raises(ValidationError):
            config.pattern_id = uuid4()  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelBaselineRunConfig(
                pattern_id=uuid4(),
                scenario_id=uuid4(),
                correlation_id=uuid4(),
                current_tier=EnumLifecycleTier.SUGGESTED,
                target_tier=EnumLifecycleTier.SHADOW_APPLY,
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ============================================================================
# ModelBaselineRunResult -- __bool__
# ============================================================================


class TestBaselineRunResultBool:
    """Tests for ModelBaselineRunResult __bool__ non-standard behaviour."""

    def test_bool_true_when_no_error(self) -> None:
        """__bool__ returns True when error field is empty."""
        result = _make_run_result(error="")
        assert bool(result) is True

    def test_bool_false_when_error_set(self) -> None:
        """__bool__ returns False when error field is non-empty."""
        result = _make_run_result(error="Something went wrong")
        assert bool(result) is False

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        result = _make_run_result()
        with pytest.raises(ValidationError):
            result.error = "mutated"  # type: ignore[misc]


# ============================================================================
# ModelCostDelta -- from_metrics
# ============================================================================


class TestCostDelta:
    """Tests for ModelCostDelta.from_metrics() delta computation."""

    def test_savings_when_candidate_uses_fewer_tokens(self) -> None:
        """Positive token_delta when candidate uses fewer tokens."""
        baseline = _make_cost(total_tokens=1000)
        candidate = _make_cost(total_tokens=600)
        delta = ModelCostDelta.from_metrics(baseline, candidate)
        assert delta.token_delta == 400
        assert delta.token_savings_pct == 40.0

    def test_overhead_when_candidate_uses_more_tokens(self) -> None:
        """Negative token_delta when candidate uses more tokens."""
        baseline = _make_cost(total_tokens=600)
        candidate = _make_cost(total_tokens=1000)
        delta = ModelCostDelta.from_metrics(baseline, candidate)
        assert delta.token_delta == -400

    def test_zero_delta_when_equal(self) -> None:
        """Zero delta when baseline and candidate have same metrics."""
        metrics = _make_cost()
        delta = ModelCostDelta.from_metrics(metrics, metrics)
        assert delta.token_delta == 0
        assert delta.prompt_token_delta == 0
        assert delta.completion_token_delta == 0
        assert delta.wall_time_delta_ms == 0.0
        assert delta.retry_delta == 0
        assert delta.token_savings_pct == 0.0
        assert delta.time_savings_pct == 0.0

    def test_time_savings(self) -> None:
        """Time savings computed correctly."""
        baseline = _make_cost(wall_time_ms=10000.0)
        candidate = _make_cost(wall_time_ms=7000.0)
        delta = ModelCostDelta.from_metrics(baseline, candidate)
        assert delta.wall_time_delta_ms == 3000.0
        assert delta.time_savings_pct == 30.0

    def test_retry_delta(self) -> None:
        """Retry delta computed correctly."""
        baseline = _make_cost(retry_count=5)
        candidate = _make_cost(retry_count=2)
        delta = ModelCostDelta.from_metrics(baseline, candidate)
        assert delta.retry_delta == 3

    def test_prompt_and_completion_deltas(self) -> None:
        """Prompt and completion deltas computed separately."""
        baseline = _make_cost(prompt_tokens=800, completion_tokens=200)
        candidate = _make_cost(prompt_tokens=500, completion_tokens=300)
        delta = ModelCostDelta.from_metrics(baseline, candidate)
        assert delta.prompt_token_delta == 300
        assert delta.completion_token_delta == -100

    def test_zero_baseline_tokens_no_division_error(self) -> None:
        """Zero baseline tokens yields 0% savings (no division by zero)."""
        baseline = _make_cost(total_tokens=0)
        candidate = _make_cost(total_tokens=100)
        delta = ModelCostDelta.from_metrics(baseline, candidate)
        assert delta.token_savings_pct == 0.0

    def test_zero_baseline_time_no_division_error(self) -> None:
        """Zero baseline time yields 0% savings (no division by zero)."""
        baseline = _make_cost(wall_time_ms=0.0)
        candidate = _make_cost(wall_time_ms=100.0)
        delta = ModelCostDelta.from_metrics(baseline, candidate)
        assert delta.time_savings_pct == 0.0

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        delta = ModelCostDelta.from_metrics(_make_cost(), _make_cost())
        with pytest.raises(ValidationError):
            delta.token_delta = 999  # type: ignore[misc]


# ============================================================================
# ModelOutcomeDelta -- from_metrics
# ============================================================================


class TestOutcomeDelta:
    """Tests for ModelOutcomeDelta.from_metrics() delta computation."""

    def test_quality_improved_candidate_passes_baseline_fails(self) -> None:
        """Quality is improved when candidate passes but baseline fails."""
        baseline = _make_outcome(passed=False, passed_checks=5)
        candidate = _make_outcome(passed=True, passed_checks=5)
        delta = ModelOutcomeDelta.from_metrics(baseline, candidate)
        assert delta.quality_improved is True

    def test_quality_improved_more_checks_passed(self) -> None:
        """Quality is improved when candidate passes more checks."""
        baseline = _make_outcome(passed=True, passed_checks=7)
        candidate = _make_outcome(passed=True, passed_checks=9)
        delta = ModelOutcomeDelta.from_metrics(baseline, candidate)
        assert delta.quality_improved is True
        assert delta.check_delta == 2

    def test_quality_improved_lower_flake_rate(self) -> None:
        """Quality is improved when same checks but lower flake rate."""
        baseline = _make_outcome(passed=True, passed_checks=9, flake_rate=0.1)
        candidate = _make_outcome(passed=True, passed_checks=9, flake_rate=0.02)
        delta = ModelOutcomeDelta.from_metrics(baseline, candidate)
        assert delta.quality_improved is True
        assert delta.flake_rate_delta > 0

    def test_quality_not_improved_when_both_pass_same(self) -> None:
        """Quality not improved when both pass with same checks and flake rate."""
        baseline = _make_outcome(passed=True, passed_checks=9, flake_rate=0.05)
        candidate = _make_outcome(passed=True, passed_checks=9, flake_rate=0.05)
        delta = ModelOutcomeDelta.from_metrics(baseline, candidate)
        assert delta.quality_improved is False

    def test_quality_not_improved_when_candidate_fails(self) -> None:
        """Quality is NOT improved when candidate fails."""
        baseline = _make_outcome(passed=True, passed_checks=9)
        candidate = _make_outcome(passed=False, passed_checks=5)
        delta = ModelOutcomeDelta.from_metrics(baseline, candidate)
        assert delta.quality_improved is False

    def test_review_iteration_delta(self) -> None:
        """Review iteration delta computed correctly."""
        baseline = _make_outcome(passed=True, review_iterations=5)
        candidate = _make_outcome(passed=True, review_iterations=2)
        delta = ModelOutcomeDelta.from_metrics(baseline, candidate)
        assert delta.review_iteration_delta == 3

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        delta = ModelOutcomeDelta.from_metrics(_make_outcome(), _make_outcome())
        with pytest.raises(ValidationError):
            delta.quality_improved = True  # type: ignore[misc]


# ============================================================================
# ModelBaselineComparisonInput -- Construction
# ============================================================================


class TestBaselineComparisonInput:
    """Tests for ModelBaselineComparisonInput construction."""

    def test_construction(self) -> None:
        """Input can be constructed with all required fields."""
        comp_input = _make_comparison_input()
        assert comp_input.config is not None
        assert comp_input.baseline_result is not None
        assert comp_input.candidate_result is not None

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        comp_input = _make_comparison_input()
        with pytest.raises(ValidationError):
            comp_input.config = _make_config()  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError, match="extra"):
            ModelBaselineComparisonInput(
                config=_make_config(),
                baseline_result=_make_run_result(variant=EnumRunVariant.BASELINE),
                candidate_result=_make_run_result(variant=EnumRunVariant.CANDIDATE),
                unknown_field="bad",  # type: ignore[call-arg]
            )


# ============================================================================
# ModelBaselineComparisonInput -- Variant Pairing Validation
# ============================================================================


class TestBaselineComparisonInputVariantValidation:
    """Tests for ModelBaselineComparisonInput variant pairing validator."""

    def test_rejects_swapped_variants(self) -> None:
        """Rejects when baseline_result has CANDIDATE variant."""
        with pytest.raises(
            ValidationError, match=r"baseline_result\.variant must be BASELINE"
        ):
            ModelBaselineComparisonInput(
                config=_make_config(),
                baseline_result=_make_run_result(variant=EnumRunVariant.CANDIDATE),
                candidate_result=_make_run_result(variant=EnumRunVariant.CANDIDATE),
            )

    def test_rejects_both_baseline_variants(self) -> None:
        """Rejects when candidate_result has BASELINE variant."""
        with pytest.raises(
            ValidationError, match=r"candidate_result\.variant must be CANDIDATE"
        ):
            ModelBaselineComparisonInput(
                config=_make_config(),
                baseline_result=_make_run_result(variant=EnumRunVariant.BASELINE),
                candidate_result=_make_run_result(variant=EnumRunVariant.BASELINE),
            )

    def test_accepts_correct_variants(self) -> None:
        """Accepts correctly paired BASELINE and CANDIDATE variants."""
        comp_input = _make_comparison_input()
        assert comp_input.baseline_result.variant == EnumRunVariant.BASELINE
        assert comp_input.candidate_result.variant == EnumRunVariant.CANDIDATE


# ============================================================================
# ModelBaselineComparisonInput -- Correlation ID Validation
# ============================================================================


class TestBaselineComparisonInputCorrelationValidation:
    """Tests for ModelBaselineComparisonInput correlation_id match validator."""

    def test_rejects_mismatched_correlation_ids(self) -> None:
        """Rejects when baseline and candidate have different correlation_ids."""
        with pytest.raises(
            ValidationError,
            match=r"results from different comparison runs cannot be paired",
        ):
            ModelBaselineComparisonInput(
                config=_make_config(),
                baseline_result=_make_run_result(
                    variant=EnumRunVariant.BASELINE,
                    correlation_id=uuid4(),
                ),
                candidate_result=_make_run_result(
                    variant=EnumRunVariant.CANDIDATE,
                    correlation_id=uuid4(),
                ),
            )

    def test_accepts_matching_correlation_ids(self) -> None:
        """Accepts when baseline and candidate share the same correlation_id."""
        cid = uuid4()
        comp_input = ModelBaselineComparisonInput(
            config=_make_config(),
            baseline_result=_make_run_result(
                variant=EnumRunVariant.BASELINE,
                correlation_id=cid,
            ),
            candidate_result=_make_run_result(
                variant=EnumRunVariant.CANDIDATE,
                correlation_id=cid,
            ),
        )
        assert comp_input.baseline_result.correlation_id == cid
        assert comp_input.candidate_result.correlation_id == cid

    def test_error_message_includes_both_ids(self) -> None:
        """Error message includes both mismatched correlation_ids for debugging."""
        baseline_cid = uuid4()
        candidate_cid = uuid4()
        with pytest.raises(ValidationError, match=str(baseline_cid)):
            ModelBaselineComparisonInput(
                config=_make_config(),
                baseline_result=_make_run_result(
                    variant=EnumRunVariant.BASELINE,
                    correlation_id=baseline_cid,
                ),
                candidate_result=_make_run_result(
                    variant=EnumRunVariant.CANDIDATE,
                    correlation_id=candidate_cid,
                ),
            )


# ============================================================================
# HandlerBaselineComparison -- Properties
# ============================================================================


class TestHandlerBaselineComparisonProperties:
    """Tests for HandlerBaselineComparison handler classification properties."""

    def test_handler_id(self) -> None:
        """handler_id returns the expected identifier."""
        handler = HandlerBaselineComparison()
        assert handler.handler_id == "handler-baseline-comparison"

    def test_handler_type(self) -> None:
        """handler_type is INFRA_HANDLER."""
        handler = HandlerBaselineComparison()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        """handler_category is NONDETERMINISTIC_COMPUTE."""
        handler = HandlerBaselineComparison()
        assert (
            handler.handler_category == EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE
        )


# ============================================================================
# HandlerBaselineComparison -- handle()
# ============================================================================


class TestHandlerBaselineComparisonHandle:
    """Tests for HandlerBaselineComparison.handle() async method."""

    @pytest.mark.asyncio
    async def test_handle_basic_comparison(self) -> None:
        """Basic comparison produces attribution record with deltas."""
        handler = HandlerBaselineComparison()
        baseline_cost = _make_cost(total_tokens=1000, wall_time_ms=5000.0)
        candidate_cost = _make_cost(total_tokens=700, wall_time_ms=3500.0)
        baseline_outcome = _make_outcome(passed=True, passed_checks=8)
        candidate_outcome = _make_outcome(passed=True, passed_checks=9)
        cid = uuid4()

        comp_input = _make_comparison_input(
            baseline=_make_run_result(
                variant=EnumRunVariant.BASELINE,
                cost=baseline_cost,
                outcome=baseline_outcome,
                correlation_id=cid,
            ),
            candidate=_make_run_result(
                variant=EnumRunVariant.CANDIDATE,
                cost=candidate_cost,
                outcome=candidate_outcome,
                correlation_id=cid,
            ),
        )

        result = await handler.handle(comp_input)

        assert isinstance(result, ModelAttributionRecord)
        assert result.cost_delta.token_delta == 300
        assert result.cost_delta.token_savings_pct == 30.0
        assert result.outcome_delta.quality_improved is True
        assert result.roi_positive is True

    @pytest.mark.asyncio
    async def test_handle_negative_roi_more_tokens(self) -> None:
        """Negative ROI when candidate uses more tokens."""
        handler = HandlerBaselineComparison()
        baseline_cost = _make_cost(total_tokens=500)
        candidate_cost = _make_cost(total_tokens=1000)
        cid = uuid4()

        comp_input = _make_comparison_input(
            baseline=_make_run_result(
                variant=EnumRunVariant.BASELINE,
                cost=baseline_cost,
                correlation_id=cid,
            ),
            candidate=_make_run_result(
                variant=EnumRunVariant.CANDIDATE,
                cost=candidate_cost,
                correlation_id=cid,
            ),
        )

        result = await handler.handle(comp_input)

        assert result.cost_delta.token_delta == -500
        assert result.roi_positive is False

    @pytest.mark.asyncio
    async def test_handle_negative_roi_candidate_fails(self) -> None:
        """Negative ROI when candidate fails validation."""
        handler = HandlerBaselineComparison()
        baseline_outcome = _make_outcome(passed=True)
        candidate_outcome = _make_outcome(passed=False)
        cid = uuid4()

        comp_input = _make_comparison_input(
            baseline=_make_run_result(
                variant=EnumRunVariant.BASELINE,
                outcome=baseline_outcome,
                correlation_id=cid,
            ),
            candidate=_make_run_result(
                variant=EnumRunVariant.CANDIDATE,
                outcome=candidate_outcome,
                correlation_id=cid,
            ),
        )

        result = await handler.handle(comp_input)

        assert result.roi_positive is False

    @pytest.mark.asyncio
    async def test_handle_preserves_correlation_id(self) -> None:
        """Result includes the correlation_id from config."""
        handler = HandlerBaselineComparison()
        config = _make_config()
        comp_input = _make_comparison_input(config=config)

        result = await handler.handle(comp_input)

        assert result.correlation_id == config.correlation_id
        assert result.pattern_id == config.pattern_id
        assert result.scenario_id == config.scenario_id

    @pytest.mark.asyncio
    async def test_handle_generates_correlation_id_when_none(self) -> None:
        """Result gets auto-generated correlation_id when config has None."""
        handler = HandlerBaselineComparison()
        config = ModelBaselineRunConfig(
            pattern_id=uuid4(),
            scenario_id=uuid4(),
            correlation_id=None,
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.SHADOW_APPLY,
        )
        comp_input = _make_comparison_input(config=config)

        result = await handler.handle(comp_input)

        assert result.correlation_id is not None

    @pytest.mark.asyncio
    async def test_handle_preserves_tier_info(self) -> None:
        """Result includes tier information from config."""
        handler = HandlerBaselineComparison()
        config = _make_config(
            current_tier=EnumLifecycleTier.SHADOW_APPLY,
            target_tier=EnumLifecycleTier.PROMOTED,
        )
        comp_input = _make_comparison_input(config=config)

        result = await handler.handle(comp_input)

        assert result.current_tier == EnumLifecycleTier.SHADOW_APPLY
        assert result.target_tier == EnumLifecycleTier.PROMOTED

    @pytest.mark.asyncio
    async def test_handle_stores_full_run_results(self) -> None:
        """Result stores the full baseline and candidate run results."""
        handler = HandlerBaselineComparison()
        cid = uuid4()
        baseline = _make_run_result(variant=EnumRunVariant.BASELINE, correlation_id=cid)
        candidate = _make_run_result(
            variant=EnumRunVariant.CANDIDATE, correlation_id=cid
        )

        comp_input = _make_comparison_input(
            baseline=baseline,
            candidate=candidate,
        )

        result = await handler.handle(comp_input)

        assert result.baseline_result == baseline
        assert result.candidate_result == candidate

    @pytest.mark.asyncio
    async def test_handle_neutral_roi_same_metrics(self) -> None:
        """Neutral comparison (same metrics) has positive ROI if both pass."""
        handler = HandlerBaselineComparison()
        cost = _make_cost()
        outcome = _make_outcome(passed=True)
        cid = uuid4()

        comp_input = _make_comparison_input(
            baseline=_make_run_result(
                variant=EnumRunVariant.BASELINE,
                cost=cost,
                outcome=outcome,
                correlation_id=cid,
            ),
            candidate=_make_run_result(
                variant=EnumRunVariant.CANDIDATE,
                cost=cost,
                outcome=outcome,
                correlation_id=cid,
            ),
        )

        result = await handler.handle(comp_input)

        # Same metrics: token_delta=0 (non-negative), candidate passed,
        # check_delta=0 (non-negative) -> ROI is positive
        assert result.cost_delta.token_delta == 0
        assert result.roi_positive is True


# ============================================================================
# HandlerBaselineComparison -- _compute_roi
# ============================================================================


class TestComputeRoi:
    """Tests for HandlerBaselineComparison._compute_roi static method."""

    def test_positive_roi_savings_and_quality(self) -> None:
        """Positive ROI: cost savings + quality improvement."""
        cost_delta = ModelCostDelta(token_delta=100, token_savings_pct=10.0)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            quality_improved=True,
        )
        assert HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is True

    def test_positive_roi_neutral_cost_quality_same(self) -> None:
        """Positive ROI: zero cost delta + candidate passed + same checks."""
        cost_delta = ModelCostDelta(token_delta=0)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            check_delta=0,
        )
        assert HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is True

    def test_negative_roi_cost_increase(self) -> None:
        """Negative ROI: cost increase (negative token delta)."""
        cost_delta = ModelCostDelta(token_delta=-100)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            quality_improved=True,
        )
        assert (
            HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is False
        )

    def test_negative_roi_candidate_failed(self) -> None:
        """Negative ROI: candidate failed validation."""
        cost_delta = ModelCostDelta(token_delta=100)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=False,
        )
        assert (
            HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is False
        )

    def test_negative_roi_quality_regression(self) -> None:
        """Negative ROI: quality regression (fewer checks passed)."""
        cost_delta = ModelCostDelta(token_delta=100)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            check_delta=-3,
            quality_improved=False,
        )
        assert (
            HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is False
        )

    def test_roi_negative_when_flake_rate_regresses(self) -> None:
        """Negative ROI: flake rate regression despite cost savings and check parity."""
        cost_delta = ModelCostDelta(token_delta=100, token_savings_pct=10.0)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            check_delta=0,
            flake_rate_delta=-0.05,
            review_iteration_delta=0,
            quality_improved=False,
        )
        assert (
            HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is False
        )

    def test_roi_negative_when_review_iterations_regress(self) -> None:
        """Negative ROI: review iteration regression despite cost savings and check parity."""
        cost_delta = ModelCostDelta(token_delta=100, token_savings_pct=10.0)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            check_delta=0,
            flake_rate_delta=0.0,
            review_iteration_delta=-2,
            quality_improved=False,
        )
        assert (
            HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is False
        )

    def test_roi_negative_when_both_flake_and_review_regress(self) -> None:
        """Negative ROI: both flake rate and review iterations regress."""
        cost_delta = ModelCostDelta(token_delta=100, token_savings_pct=10.0)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            check_delta=0,
            flake_rate_delta=-0.1,
            review_iteration_delta=-3,
            quality_improved=False,
        )
        assert (
            HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is False
        )

    def test_roi_negative_when_quality_improved_but_flake_regresses(self) -> None:
        """Negative ROI: quality_improved=True must not mask flake/review regressions."""
        cost_delta = ModelCostDelta(token_delta=100, token_savings_pct=10.0)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            check_delta=2,
            flake_rate_delta=-0.05,
            review_iteration_delta=-1,
            quality_improved=True,
        )
        assert (
            HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is False
        )

    def test_roi_positive_when_quality_improved_and_no_regressions(self) -> None:
        """Positive ROI: quality_improved=True with no flake/review regressions."""
        cost_delta = ModelCostDelta(token_delta=100, token_savings_pct=10.0)
        outcome_delta = ModelOutcomeDelta(
            baseline_passed=True,
            candidate_passed=True,
            check_delta=2,
            flake_rate_delta=0.0,
            review_iteration_delta=0,
            quality_improved=True,
        )
        assert HandlerBaselineComparison._compute_roi(cost_delta, outcome_delta) is True


# ============================================================================
# ModelAttributionRecord -- Construction
# ============================================================================


class TestAttributionRecord:
    """Tests for ModelAttributionRecord construction."""

    def test_construction(self) -> None:
        """Full attribution record can be constructed."""
        now = datetime.now(tz=UTC)
        record = ModelAttributionRecord(
            record_id=uuid4(),
            pattern_id=uuid4(),
            scenario_id=uuid4(),
            correlation_id=uuid4(),
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.SHADOW_APPLY,
            baseline_result=_make_run_result(variant=EnumRunVariant.BASELINE),
            candidate_result=_make_run_result(variant=EnumRunVariant.CANDIDATE),
            cost_delta=ModelCostDelta(token_delta=100),
            outcome_delta=ModelOutcomeDelta(
                baseline_passed=True, candidate_passed=True
            ),
            roi_positive=True,
            created_at=now,
        )
        assert record.roi_positive is True
        assert record.current_tier == EnumLifecycleTier.SUGGESTED

    def test_frozen_immutability(self) -> None:
        """Frozen model rejects attribute mutation."""
        now = datetime.now(tz=UTC)
        record = ModelAttributionRecord(
            record_id=uuid4(),
            pattern_id=uuid4(),
            scenario_id=uuid4(),
            correlation_id=uuid4(),
            current_tier=EnumLifecycleTier.SUGGESTED,
            target_tier=EnumLifecycleTier.SHADOW_APPLY,
            baseline_result=_make_run_result(variant=EnumRunVariant.BASELINE),
            candidate_result=_make_run_result(variant=EnumRunVariant.CANDIDATE),
            cost_delta=ModelCostDelta(token_delta=100),
            outcome_delta=ModelOutcomeDelta(
                baseline_passed=True, candidate_passed=True
            ),
            roi_positive=True,
            created_at=now,
        )
        with pytest.raises(ValidationError):
            record.roi_positive = False  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Extra fields are rejected."""
        now = datetime.now(tz=UTC)
        with pytest.raises(ValidationError, match="extra"):
            ModelAttributionRecord(
                record_id=uuid4(),
                pattern_id=uuid4(),
                scenario_id=uuid4(),
                correlation_id=uuid4(),
                current_tier=EnumLifecycleTier.SUGGESTED,
                target_tier=EnumLifecycleTier.SHADOW_APPLY,
                baseline_result=_make_run_result(variant=EnumRunVariant.BASELINE),
                candidate_result=_make_run_result(variant=EnumRunVariant.CANDIDATE),
                cost_delta=ModelCostDelta(token_delta=100),
                outcome_delta=ModelOutcomeDelta(
                    baseline_passed=True, candidate_passed=True
                ),
                roi_positive=True,
                created_at=now,
                unknown_field="bad",  # type: ignore[call-arg]
            )
