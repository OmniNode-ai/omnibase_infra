# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for eval event emitter [OMN-6779]."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from onex_change_control.models.model_eval_report import (
    ModelEvalReport,
    ModelEvalSummary,
)

from omnibase_infra.services.eval.eval_event_emitter import (
    EVAL_COMPLETED_TOPIC,
    build_eval_completed_payload,
    serialize_eval_event,
)


@pytest.fixture
def sample_report() -> ModelEvalReport:
    return ModelEvalReport(
        report_id="test-report-001",
        suite_id="test-suite",
        suite_version="1.0.0",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        pairs=[],
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
