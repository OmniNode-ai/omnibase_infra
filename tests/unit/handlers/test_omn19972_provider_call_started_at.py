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
