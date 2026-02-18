# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for baselines observability models.

Tests Pydantic model construction, validation, and frozen semantics
for all baselines row models.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit

from omnibase_infra.services.observability.baselines.models.model_baselines_breakdown_row import (
    ModelBaselinesBreakdownRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_comparison_row import (
    ModelBaselinesComparisonRow,
)
from omnibase_infra.services.observability.baselines.models.model_baselines_trend_row import (
    ModelBaselinesTrendRow,
)

# ============================================================================
# Helpers
# ============================================================================


def _now() -> datetime:
    return datetime.now(UTC)


def _make_comparison_row(**overrides: object) -> ModelBaselinesComparisonRow:
    defaults: dict[str, object] = {
        "id": uuid4(),
        "comparison_date": date(2026, 2, 18),
        "treatment_sessions": 100,
        "treatment_success_rate": 0.85,
        "treatment_avg_latency_ms": 200.0,
        "treatment_avg_cost_tokens": 1500.0,
        "treatment_total_tokens": 150000,
        "control_sessions": 80,
        "control_success_rate": 0.70,
        "control_avg_latency_ms": 260.0,
        "control_avg_cost_tokens": 1800.0,
        "control_total_tokens": 144000,
        "roi_pct": 21.4,
        "latency_improvement_pct": 23.1,
        "cost_improvement_pct": 16.7,
        "sample_size": 180,
        "computed_at": _now(),
        "created_at": _now(),
        "updated_at": _now(),
    }
    defaults.update(overrides)
    return ModelBaselinesComparisonRow(**defaults)  # type: ignore[arg-type]


def _make_trend_row(**overrides: object) -> ModelBaselinesTrendRow:
    defaults: dict[str, object] = {
        "id": uuid4(),
        "trend_date": date(2026, 2, 18),
        "cohort": "treatment",
        "session_count": 50,
        "success_rate": 0.82,
        "avg_latency_ms": 180.0,
        "avg_cost_tokens": 1200.0,
        "roi_pct": 17.1,
        "computed_at": _now(),
        "created_at": _now(),
    }
    defaults.update(overrides)
    return ModelBaselinesTrendRow(**defaults)  # type: ignore[arg-type]


def _make_breakdown_row(**overrides: object) -> ModelBaselinesBreakdownRow:
    defaults: dict[str, object] = {
        "id": uuid4(),
        "pattern_id": uuid4(),
        "pattern_label": "coder-agent",
        "treatment_success_rate": 0.88,
        "control_success_rate": 0.72,
        "roi_pct": 22.2,
        "sample_count": 45,
        "treatment_count": 30,
        "control_count": 15,
        "confidence": 0.88,
        "computed_at": _now(),
        "created_at": _now(),
        "updated_at": _now(),
    }
    defaults.update(overrides)
    return ModelBaselinesBreakdownRow(**defaults)  # type: ignore[arg-type]


# ============================================================================
# ModelBaselinesComparisonRow
# ============================================================================


class TestModelBaselinesComparisonRow:
    """Tests for ModelBaselinesComparisonRow."""

    def test_construction_full(self) -> None:
        row = _make_comparison_row()
        assert row.comparison_date == date(2026, 2, 18)
        assert row.treatment_sessions == 100
        assert row.control_sessions == 80
        assert row.roi_pct == pytest.approx(21.4)

    def test_optional_fields_default_none(self) -> None:
        row = _make_comparison_row(
            treatment_success_rate=None,
            control_success_rate=None,
            roi_pct=None,
            latency_improvement_pct=None,
            cost_improvement_pct=None,
            period_label=None,
        )
        assert row.treatment_success_rate is None
        assert row.control_success_rate is None
        assert row.roi_pct is None
        assert row.period_label is None

    def test_frozen_model_rejects_mutation(self) -> None:
        row = _make_comparison_row()
        with pytest.raises(ValidationError):
            row.roi_pct = 99.9  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            _make_comparison_row(unknown_field="bad")  # type: ignore[call-arg]

    def test_negative_session_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_comparison_row(treatment_sessions=-1)

    def test_negative_total_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_comparison_row(treatment_total_tokens=-100)

    def test_period_label_optional(self) -> None:
        row = _make_comparison_row(period_label="Q1 2026")
        assert row.period_label == "Q1 2026"


# ============================================================================
# ModelBaselinesTrendRow
# ============================================================================


class TestModelBaselinesTrendRow:
    """Tests for ModelBaselinesTrendRow."""

    def test_construction_treatment_cohort(self) -> None:
        row = _make_trend_row(cohort="treatment")
        assert row.cohort == "treatment"
        assert row.session_count == 50

    def test_construction_control_cohort(self) -> None:
        row = _make_trend_row(cohort="control", roi_pct=None)
        assert row.cohort == "control"
        assert row.roi_pct is None

    def test_optional_metrics_can_be_none(self) -> None:
        row = _make_trend_row(
            success_rate=None,
            avg_latency_ms=None,
            avg_cost_tokens=None,
            roi_pct=None,
        )
        assert row.success_rate is None
        assert row.avg_latency_ms is None

    def test_frozen_model_rejects_mutation(self) -> None:
        row = _make_trend_row()
        with pytest.raises(ValidationError):
            row.session_count = 0  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            _make_trend_row(unknown="bad")  # type: ignore[call-arg]

    def test_negative_session_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_trend_row(session_count=-5)


# ============================================================================
# ModelBaselinesBreakdownRow
# ============================================================================


class TestModelBaselinesBreakdownRow:
    """Tests for ModelBaselinesBreakdownRow."""

    def test_construction_full(self) -> None:
        row = _make_breakdown_row()
        assert row.pattern_label == "coder-agent"
        assert row.sample_count == 45
        assert row.confidence == pytest.approx(0.88)

    def test_confidence_can_be_none(self) -> None:
        row = _make_breakdown_row(confidence=None, sample_count=5)
        assert row.confidence is None

    def test_optional_rates_can_be_none(self) -> None:
        row = _make_breakdown_row(
            treatment_success_rate=None,
            control_success_rate=None,
            roi_pct=None,
        )
        assert row.treatment_success_rate is None
        assert row.roi_pct is None

    def test_frozen_model_rejects_mutation(self) -> None:
        row = _make_breakdown_row()
        with pytest.raises(ValidationError):
            row.roi_pct = 100.0  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            _make_breakdown_row(extra_field="bad")  # type: ignore[call-arg]

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_breakdown_row(sample_count=-1)

        with pytest.raises(ValidationError):
            _make_breakdown_row(treatment_count=-1)

        with pytest.raises(ValidationError):
            _make_breakdown_row(control_count=-1)

    def test_pattern_label_optional(self) -> None:
        row = _make_breakdown_row(pattern_label=None)
        assert row.pattern_label is None
