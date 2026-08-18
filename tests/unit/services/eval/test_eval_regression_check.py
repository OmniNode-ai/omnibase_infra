# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for eval regression check [OMN-6782]."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_core.enums.governance.enum_eval_mode import EnumEvalMode
from omnibase_core.enums.governance.enum_eval_verdict import EnumEvalVerdict
from omnibase_core.models.governance.model_eval_report import ModelEvalReport
from omnibase_core.models.governance.model_eval_run import ModelEvalRun
from omnibase_core.models.governance.model_eval_run_pair import ModelEvalRunPair
from omnibase_core.models.governance.model_eval_summary import ModelEvalSummary
from omnibase_infra.services.eval.eval_regression_check import (
    check_eval_regression,
)


def _make_run(task_id: str, mode: EnumEvalMode) -> ModelEvalRun:
    return ModelEvalRun(
        run_id=f"run-{task_id}-{mode.value}",
        task_id=task_id,
        mode=mode,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        success=True,
        git_sha="abc123",
    )


def _make_pairs(total: int) -> list[ModelEvalRunPair]:
    """Build `total` minimal, unique-task_id pairs to satisfy
    ModelEvalReport.validate_summary_alignment (summary.total_tasks == len(pairs)).
    """
    return [
        ModelEvalRunPair(
            task_id=f"task-{i}",
            onex_on_run=_make_run(f"task-{i}", EnumEvalMode.ONEX_ON),
            onex_off_run=_make_run(f"task-{i}", EnumEvalMode.ONEX_OFF),
            verdict=EnumEvalVerdict.NEUTRAL,
        )
        for i in range(total)
    ]


def _make_report(
    total: int = 10,
    better: int | None = None,
    worse: int = 2,
    neutral: int = 2,
) -> ModelEvalReport:
    """Build a report whose summary counts satisfy
    ModelEvalSummary.validate_counts (better + worse + neutral == total).

    `better` defaults to the remainder of `total` after `worse`/`neutral`
    so callers that only vary `worse` (as most tests here do) still get a
    self-consistent summary.
    """
    if better is None:
        better = total - worse - neutral
    return ModelEvalReport(
        report_id="test-report",
        suite_id="test-suite",
        suite_version="1.0.0",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        pairs=_make_pairs(total),
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

    def test_threshold_above_one_raises(self) -> None:
        report = _make_report()
        with pytest.raises(ValueError, match="threshold must be in"):
            check_eval_regression(report, threshold=1.5)

    def test_negative_threshold_raises(self) -> None:
        report = _make_report()
        with pytest.raises(ValueError, match="threshold must be in"):
            check_eval_regression(report, threshold=-0.1)
