# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for post-recreate container start verification (OMN-8668).

Covers the scenario where docker compose up exits 0 but containers land in
Created state instead of running — the bug that caused silent infra outages
at 01:33 and 04:48 on 2026-04-13.
"""

from __future__ import annotations

import subprocess
from unittest.mock import call, patch

import pytest
from deploy_agent.executor import verify_containers_up


def _make_ps_result(
    name_state_pairs: list[tuple[str, str]],
) -> subprocess.CompletedProcess:
    stdout = "\n".join(f"{name}\t{state}" for name, state in name_state_pairs)
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_all_running_returns_true() -> None:
    """Returns (True, []) immediately when all containers are running."""
    containers = ["postgres", "redpanda", "valkey"]
    ps_output = _make_ps_result([(c, "running") for c in containers])

    with patch(
        "deploy_agent.executor.subprocess.run", return_value=ps_output
    ) as mock_run:
        ok, stuck = verify_containers_up(containers, timeout_s=10)

    assert ok is True
    assert stuck == []
    assert mock_run.call_count == 1


def test_created_state_detected_as_missing() -> None:
    """Containers in Created state are treated as not running (the OMN-8668 bug)."""
    containers = ["postgres", "redpanda"]
    # redpanda stuck in Created — the exact failure mode
    ps_output = _make_ps_result([("postgres", "running"), ("redpanda", "created")])

    call_count = 0

    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess:
        nonlocal call_count
        call_count += 1
        return ps_output

    with patch("deploy_agent.executor.subprocess.run", side_effect=fake_run):
        with patch("deploy_agent.executor.time.sleep"):
            with patch("deploy_agent.executor.time.monotonic", side_effect=[0, 0, 200]):
                ok, stuck = verify_containers_up(containers, timeout_s=1)

    assert ok is False
    assert "redpanda" in stuck
    assert "postgres" not in stuck


def test_recovery_on_second_poll() -> None:
    """Returns (True, []) when container transitions to running on second poll."""
    containers = ["postgres", "redpanda"]
    first_call = _make_ps_result([("postgres", "running"), ("redpanda", "created")])
    second_call = _make_ps_result([("postgres", "running"), ("redpanda", "running")])

    responses = [first_call, second_call]

    with patch("deploy_agent.executor.subprocess.run", side_effect=responses):
        with patch("deploy_agent.executor.time.sleep"):
            ok, stuck = verify_containers_up(containers, timeout_s=60)

    assert ok is True
    assert stuck == []


def test_docker_ps_failure_skips_iteration() -> None:
    """When docker ps returns non-zero, loop retries rather than crashing."""
    containers = ["postgres"]
    fail_result = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="docker daemon unavailable"
    )
    ok_result = _make_ps_result([("postgres", "running")])

    with patch(
        "deploy_agent.executor.subprocess.run", side_effect=[fail_result, ok_result]
    ):
        with patch("deploy_agent.executor.time.sleep"):
            ok, stuck = verify_containers_up(containers, timeout_s=60)

    assert ok is True
    assert stuck == []


def test_empty_expected_list_returns_true() -> None:
    """Empty expected list means nothing to wait for — return True immediately."""
    ps_output = _make_ps_result([])

    with patch("deploy_agent.executor.subprocess.run", return_value=ps_output):
        ok, stuck = verify_containers_up([], timeout_s=10)

    assert ok is True
    assert stuck == []


def test_timeout_returns_false_with_stuck_list() -> None:
    """When timeout expires, returns (False, <stuck containers>)."""
    containers = ["postgres", "omninode-runtime"]
    stuck_output = _make_ps_result(
        [("postgres", "running"), ("omninode-runtime", "created")]
    )

    monotonic_values = [0.0, 0.5, 1.5]  # deadline=1, third call is past deadline

    with patch("deploy_agent.executor.subprocess.run", return_value=stuck_output):
        with patch("deploy_agent.executor.time.sleep"):
            with patch(
                "deploy_agent.executor.time.monotonic", side_effect=monotonic_values
            ):
                ok, stuck = verify_containers_up(containers, timeout_s=1)

    assert ok is False
    assert "omninode-runtime" in stuck
    assert "postgres" not in stuck
