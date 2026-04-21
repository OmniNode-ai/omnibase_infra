# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for PublishCircuitBreaker."""

from __future__ import annotations

import time

import pytest
from deploy_agent.publisher import PublishCircuitBreaker


def test_not_tripped_initially() -> None:
    cb = PublishCircuitBreaker()
    assert not cb.is_tripped("job-1")


def test_trips_after_max_consecutive_failures() -> None:
    cb = PublishCircuitBreaker(max_consecutive_failures=3, max_age_seconds=3600.0)
    for _ in range(3):
        cb.record_failure("job-1")
    assert cb.is_tripped("job-1")


def test_not_tripped_below_threshold() -> None:
    cb = PublishCircuitBreaker(max_consecutive_failures=5, max_age_seconds=3600.0)
    for _ in range(4):
        cb.record_failure("job-1")
    assert not cb.is_tripped("job-1")


def test_trips_after_max_age(monkeypatch: pytest.MonkeyPatch) -> None:
    cb = PublishCircuitBreaker(max_consecutive_failures=100, max_age_seconds=10.0)
    cb.record_failure("job-1")

    # Advance monotonic time past the age threshold
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + 11.0)

    assert cb.is_tripped("job-1")


def test_success_clears_state() -> None:
    cb = PublishCircuitBreaker(max_consecutive_failures=3, max_age_seconds=3600.0)
    for _ in range(3):
        cb.record_failure("job-1")
    assert cb.is_tripped("job-1")

    cb.record_success("job-1")
    assert not cb.is_tripped("job-1")


def test_clear_removes_state() -> None:
    cb = PublishCircuitBreaker(max_consecutive_failures=3, max_age_seconds=3600.0)
    for _ in range(3):
        cb.record_failure("job-1")
    cb.clear("job-1")
    assert not cb.is_tripped("job-1")


def test_independent_jobs_tracked_separately() -> None:
    cb = PublishCircuitBreaker(max_consecutive_failures=3, max_age_seconds=3600.0)
    for _ in range(3):
        cb.record_failure("job-A")
    cb.record_failure("job-B")

    assert cb.is_tripped("job-A")
    assert not cb.is_tripped("job-B")


def test_success_on_unknown_job_is_noop() -> None:
    cb = PublishCircuitBreaker()
    cb.record_success("nonexistent")  # must not raise
    assert not cb.is_tripped("nonexistent")
