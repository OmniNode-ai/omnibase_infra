# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for eval regression check [OMN-6782]."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from onex_change_control.models.model_eval_report import (
    ModelEvalReport,
    ModelEvalSummary,
)

from omnibase_infra.services.eval.eval_regression_check import (
    check_eval_regression,
)


def _make_report(
    total: int = 10,
    better: int = 6,
    worse: int = 2,
    neutral: int = 2,
) -> ModelEvalReport:
    return ModelEvalReport(
        report_id="test-report",
        suite_id="test-suite",
        suite_version="1.0.0",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        pairs=[],
        summary=ModelEvalSummary(
            total_tasks=total,
            onex_better_count=better,
            onex_worse_count=worse,
            neutral_count=neutral,
            avg_latency_delta_ms=0.0,
            avg_token_delta=0.0,
            avg_success_rate_on=0.8,
            avg_success_rate_off=0.7,
            pattern_hit_rate_on=0.5,
        ),
    )


@pytest.mark.unit
class TestCheckEvalRegression:
    def test_no_regression_below_threshold(self) -> None:
        report = _make_report(total=10, worse=2)
        result = check_eval_regression(report, threshold=0.30)
        assert result.is_regression is False
        assert result.worse_ratio == 0.2

    def test_regression_above_threshold(self) -> None:
        report = _make_report(total=10, worse=4)
        result = check_eval_regression(report, threshold=0.30)
        assert result.is_regression is True
        assert result.worse_ratio == 0.4

    def test_boundary_at_threshold(self) -> None:
        report = _make_report(total=10, worse=3)
        result = check_eval_regression(report, threshold=0.30)
        # 3/10 = 0.30, not > 0.30
        assert result.is_regression is False

    def test_empty_report(self) -> None:
        report = _make_report(total=0, better=0, worse=0, neutral=0)
        result = check_eval_regression(report)
        assert result.is_regression is False
        assert result.total_tasks == 0

    def test_summary_message(self) -> None:
        report = _make_report(total=10, worse=4)
        result = check_eval_regression(report, threshold=0.30)
        assert "REGRESSION" in result.summary

    def test_ok_summary_message(self) -> None:
        report = _make_report(total=10, worse=1)
        result = check_eval_regression(report, threshold=0.30)
        assert "OK" in result.summary
