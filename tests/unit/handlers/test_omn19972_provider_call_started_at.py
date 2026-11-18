# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Review finding of OMN-19972: a provider_call step records when that call started, not when the run did."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.handlers.handler_delegation_dispatch_terminal import (
    RunStepSequence,
)

pytestmark = pytest.mark.unit


def _sequence(run_started_at: datetime) -> RunStepSequence:
    return RunStepSequence(
        tenant_id="tenant-a",
        correlation_id=uuid4(),
        started_at=run_started_at,
        started=time.monotonic(),
    )


def test_a_provider_call_starts_when_the_call_started_not_when_the_run_did() -> None:
    run_started_at = datetime.now(UTC) - timedelta(minutes=10)
    sequence = _sequence(run_started_at)
    before = datetime.now(UTC)
    sequence.provider_call(
        duration_ms=2000,
        backend_id="backend-a",
        model_id="model-a",
        failure_cause=None,
        reason=None,
    )
    after = datetime.now(UTC)
    (step,) = sequence.steps
    assert step.started_at != run_started_at
    assert before - timedelta(milliseconds=2000) <= step.started_at
    assert step.started_at <= after - timedelta(milliseconds=2000)


def test_two_attempts_do_not_share_the_run_start() -> None:
    run_started_at = datetime.now(UTC) - timedelta(minutes=10)
    sequence = _sequence(run_started_at)
    for _ in range(2):
        sequence.provider_call(
            duration_ms=10,
            backend_id="backend-a",
            model_id="model-a",
            failure_cause=None,
            reason=None,
        )
    assert all(step.started_at != run_started_at for step in sequence.steps)


def test_the_request_step_still_carries_the_run_start() -> None:
    """Control: the request step is the run's own start."""
    run_started_at = datetime.now(UTC) - timedelta(minutes=10)
    sequence = _sequence(run_started_at)
    sequence.request(task_type="code_review")
    (step,) = sequence.steps
    assert step.started_at == run_started_at


def test_a_routing_decision_is_stamped_before_the_call_it_chose() -> None:
    """Review finding of OMN-19972: the routing step is not stamped after the call ran."""
    sequence = _sequence(datetime.now(UTC) - timedelta(minutes=10))
    sequence.routing(backend_id="backend-a", model_id="model-a", tier="local")
    sequence.provider_call(
        duration_ms=5000,
        backend_id="backend-a",
        model_id="model-a",
        failure_cause=None,
        reason=None,
    )
    routing, call = sequence.steps
    assert routing.step_kind == "routing_decision"
    assert routing.started_at <= call.started_at


def test_a_second_attempt_routing_decision_precedes_its_own_call() -> None:
    sequence = _sequence(datetime.now(UTC) - timedelta(minutes=10))
    for backend in ("backend-a", "backend-b"):
        sequence.routing(backend_id=backend, model_id="m", tier="local")
        sequence.provider_call(
            duration_ms=3000,
            backend_id=backend,
            model_id="m",
            failure_cause=None,
            reason=None,
        )
    first_route, first_call, second_route, second_call = sequence.steps
    assert first_route.started_at <= first_call.started_at
    assert second_route.started_at <= second_call.started_at


def test_a_routing_decision_without_a_call_keeps_its_own_stamp() -> None:
    """Control: a decision no call follows is not moved."""
    sequence = _sequence(datetime.now(UTC) - timedelta(minutes=10))
    before = datetime.now(UTC)
    sequence.routing(backend_id="backend-a", model_id="model-a", tier="local")
    (routing,) = sequence.steps
    assert routing.started_at >= before


def test_a_quality_gate_reports_no_duration_it_never_measured() -> None:
    sequence = _sequence(datetime.now(UTC) - timedelta(minutes=10))
    sequence.provider_call(
        duration_ms=10,
        backend_id="backend-a",
        model_id="model-a",
        failure_cause=None,
        reason=None,
    )
    sequence.quality_gate(backend_id="backend-a", failure_cause=None, reason=None)
    gate = sequence.steps[-1]
    assert gate.step_kind == "quality_gate"
    assert gate.duration_ms is None
