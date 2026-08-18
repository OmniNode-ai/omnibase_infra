# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for eval event emitter [OMN-6779]."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from omnibase_core.enums.governance.enum_eval_mode import EnumEvalMode
from omnibase_core.enums.governance.enum_eval_verdict import EnumEvalVerdict
from omnibase_core.models.governance.model_eval_report import ModelEvalReport
from omnibase_core.models.governance.model_eval_run import ModelEvalRun
from omnibase_core.models.governance.model_eval_run_pair import ModelEvalRunPair
from omnibase_core.models.governance.model_eval_summary import ModelEvalSummary
from omnibase_infra.services.eval.eval_event_emitter import (
    EVAL_COMPLETED_TOPIC,
    build_eval_completed_payload,
    serialize_eval_event,
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


@pytest.fixture
def sample_report() -> ModelEvalReport:
    return ModelEvalReport(
        report_id="test-report-001",
        suite_id="test-suite",
        suite_version="1.0.0",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        pairs=_make_pairs(10),
        summary=ModelEvalSummary(
            total_tasks=10,
            onex_better_count=6,
            onex_worse_count=2,
            neutral_count=2,
            avg_latency_delta_ms=-50.0,
            avg_token_delta=-100.0,
            avg_success_rate_on=0.9,
            avg_success_rate_off=0.7,
            pattern_hit_rate_on=0.8,
        ),
    )


@pytest.mark.unit
class TestBuildPayload:
    def test_contains_required_fields(self, sample_report: ModelEvalReport) -> None:
        payload = build_eval_completed_payload(sample_report)
        assert payload["event_type"] == "eval_completed"
        assert payload["topic"] == EVAL_COMPLETED_TOPIC
        assert payload["report_id"] == "test-report-001"
        assert payload["suite_id"] == "test-suite"
        assert "summary" in payload

    def test_summary_fields(self, sample_report: ModelEvalReport) -> None:
        payload = build_eval_completed_payload(sample_report)
        summary = payload["summary"]
        assert isinstance(summary, dict)
        assert summary["total_tasks"] == 10
        assert summary["onex_better_count"] == 6


@pytest.mark.unit
class TestSerializeEvent:
    def test_returns_bytes(self, sample_report: ModelEvalReport) -> None:
        data = serialize_eval_event(sample_report)
        assert isinstance(data, bytes)
        parsed = json.loads(data)
        assert parsed["report_id"] == "test-report-001"
