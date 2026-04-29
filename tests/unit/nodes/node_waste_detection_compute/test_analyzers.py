# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for waste detection analyzer rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_agent_loop import (
    analyze_agent_loop,
)
from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_high_output import (
    analyze_high_output,
)
from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_low_cache import (
    analyze_low_cache,
)
from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_model_overkill import (
    analyze_model_overkill,
)
from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_retry_waste import (
    analyze_retry_waste,
)
from omnibase_infra.nodes.node_waste_detection_compute.handlers.analyzer_tool_failure_waste import (
    analyze_tool_failure_waste,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import ModelWasteCall

BASE_TIME = datetime(2026, 4, 29, 12, 0, 0, tzinfo=UTC)
DETECTED_AT = datetime(2026, 4, 29, 12, 1, 0, tzinfo=UTC)


def _call(**overrides: object) -> ModelWasteCall:
    values: dict[str, object] = {
        "session_id": "sess-test",
        "model_id": "claude-sonnet-4-5",
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cost_usd": 0.003,
        "request_type": "completion",
        "emitted_at": BASE_TIME,
        "correlation_id": "corr-base",
        "repo_name": "omnibase_infra",
        "machine_id": "devbox-1",
        "status": "success",
    }
    values.update(overrides)
    return ModelWasteCall(**values)  # type: ignore[arg-type]


@pytest.mark.unit
def test_tool_failure_waste_rule_detects_failed_calls() -> None:
    findings = analyze_tool_failure_waste(
        (
            _call(
                correlation_id="failed-1",
                total_tokens=500,
                input_tokens=400,
                output_tokens=100,
                cost_usd=0.02,
                status="failed",
                error_type="tool_exit_1",
            ),
        ),
        DETECTED_AT,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "tool_failure_waste"
    assert finding.severity == "MEDIUM"
    assert finding.waste_tokens == 500
    assert finding.waste_cost_usd == 0.02


@pytest.mark.unit
def test_agent_loop_rule_detects_repeated_action_window() -> None:
    calls = tuple(
        _call(
            correlation_id=f"loop-{idx}",
            tool_name="bash",
            tool_input_hash="same-command",
            status="looping",
            emitted_at=BASE_TIME + timedelta(seconds=offset),
            total_tokens=100,
            input_tokens=80,
            output_tokens=20,
            cost_usd=0.001,
        )
        for idx, offset in enumerate((0, 5, 8), start=1)
    )

    findings = analyze_agent_loop(calls, DETECTED_AT)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "agent_loop"
    assert finding.severity == "MEDIUM"
    assert finding.waste_tokens == 200
    assert finding.waste_cost_usd == 0.002


@pytest.mark.unit
def test_retry_waste_rule_detects_duplicate_successful_request() -> None:
    calls = (
        _call(
            correlation_id="retry-first",
            tool_name="read",
            tool_input_hash="same-file",
            emitted_at=BASE_TIME,
            total_tokens=600,
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.012,
        ),
        _call(
            correlation_id="retry-second",
            tool_name="read",
            tool_input_hash="same-file",
            emitted_at=BASE_TIME + timedelta(seconds=30),
            total_tokens=600,
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.012,
        ),
    )

    findings = analyze_retry_waste(calls, DETECTED_AT)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "retry_waste"
    assert finding.severity == "MEDIUM"
    assert finding.waste_tokens == 600
    assert finding.waste_cost_usd == 0.012


@pytest.mark.unit
def test_high_output_rule_detects_excess_completion_tokens() -> None:
    findings = analyze_high_output(
        (
            _call(
                correlation_id="high-output",
                input_tokens=500,
                output_tokens=5000,
                total_tokens=5500,
                cost_usd=0.11,
            ),
        ),
        DETECTED_AT,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "high_output"
    assert finding.severity == "MEDIUM"
    assert finding.waste_tokens == 4000
    assert finding.waste_cost_usd == 0.08


@pytest.mark.unit
def test_model_overkill_rule_detects_premium_model_on_simple_work() -> None:
    findings = analyze_model_overkill(
        (
            _call(
                correlation_id="overkill",
                model_id="claude-opus-4-6",
                request_type="classification",
                input_tokens=300,
                output_tokens=100,
                total_tokens=400,
                cost_usd=0.03,
            ),
        ),
        DETECTED_AT,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "model_overkill"
    assert finding.severity == "LOW"
    assert finding.waste_tokens == 400
    assert finding.waste_cost_usd == 0.015


@pytest.mark.unit
def test_low_cache_rule_detects_large_uncached_prompt() -> None:
    findings = analyze_low_cache(
        (
            _call(
                correlation_id="low-cache",
                input_tokens=10000,
                output_tokens=500,
                total_tokens=10500,
                cost_usd=0.105,
                cache_read_tokens=100,
            ),
        ),
        DETECTED_AT,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.rule_id == "low_cache"
    assert finding.severity == "MEDIUM"
    assert finding.waste_tokens == 4900
    assert finding.waste_cost_usd == 0.049
