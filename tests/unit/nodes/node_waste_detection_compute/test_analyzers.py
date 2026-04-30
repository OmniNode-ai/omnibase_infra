# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for waste detection analyzer rules."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_agent_loop import (
    analyze_agent_loop,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_high_output import (
    analyze_high_output,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_low_cache import (
    analyze_low_cache,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_model_overkill import (
    analyze_model_overkill,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_retry_waste import (
    analyze_retry_waste,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_tool_failure_waste import (
    analyze_tool_failure_waste,
)
from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_utils import (
    cost_for_tokens,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteFinding,
)

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
def test_agent_loop_rule_counts_full_burst_window() -> None:
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
        for idx, offset in enumerate((0, 2, 4, 6, 8), start=1)
    )

    findings = analyze_agent_loop(calls, DETECTED_AT)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "HIGH"
    assert finding.waste_tokens == 400
    assert finding.evidence["repeat_count"] == 5


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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("analyzer", "calls"),
    [
        (
            analyze_high_output,
            (
                _call(
                    correlation_id="high-output-b",
                    emitted_at=BASE_TIME + timedelta(seconds=1),
                    input_tokens=500,
                    output_tokens=5000,
                    total_tokens=5500,
                    cost_usd=0.11,
                ),
                _call(
                    correlation_id="high-output-a",
                    emitted_at=BASE_TIME,
                    input_tokens=500,
                    output_tokens=4500,
                    total_tokens=5000,
                    cost_usd=0.10,
                ),
            ),
        ),
        (
            analyze_model_overkill,
            (
                _call(
                    correlation_id="overkill-b",
                    emitted_at=BASE_TIME + timedelta(seconds=1),
                    model_id="claude-opus-4-6",
                    request_type="classification",
                    total_tokens=400,
                    cost_usd=0.03,
                ),
                _call(
                    correlation_id="overkill-a",
                    emitted_at=BASE_TIME,
                    model_id="gpt-4.1",
                    request_type="routing",
                    total_tokens=300,
                    cost_usd=0.02,
                ),
            ),
        ),
        (
            analyze_low_cache,
            (
                _call(
                    correlation_id="low-cache-b",
                    emitted_at=BASE_TIME + timedelta(seconds=1),
                    input_tokens=10000,
                    output_tokens=500,
                    total_tokens=10500,
                    cost_usd=0.105,
                    cache_read_tokens=100,
                ),
                _call(
                    correlation_id="low-cache-a",
                    emitted_at=BASE_TIME,
                    input_tokens=9000,
                    output_tokens=500,
                    total_tokens=9500,
                    cost_usd=0.095,
                    cache_read_tokens=50,
                ),
            ),
        ),
    ],
)
def test_order_stable_analyzer_evidence_hashes(
    analyzer: Callable[
        [tuple[ModelWasteCall, ...], datetime], tuple[ModelWasteFinding, ...]
    ],
    calls: tuple[ModelWasteCall, ...],
) -> None:
    forward = analyzer(calls, DETECTED_AT)[0]
    reversed_order = analyzer(tuple(reversed(calls)), DETECTED_AT)[0]

    assert forward.evidence_hash == reversed_order.evidence_hash
    assert forward.dedup_key == reversed_order.dedup_key


@pytest.mark.unit
def test_cost_for_tokens_clamps_non_positive_tokens() -> None:
    call = _call(total_tokens=100, cost_usd=0.10)

    assert cost_for_tokens(call, -10) == 0.0
    assert cost_for_tokens(call, 0) == 0.0
