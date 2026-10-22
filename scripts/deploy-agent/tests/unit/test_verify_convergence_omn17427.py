# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the probe sequences reported for dev jobs c05a4116 and 1a0f5785.

The brief supplies observations rather than raw journal exports: transient 503
after the gateway while Docker is healthy, and readiness after the old 600 s
bound while still inside the declared 1800 s start period. These are modeled
sequences, not purported verbatim logs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.executor import DeployExecutor, VerificationFailedError

pytestmark = pytest.mark.unit


class _Convergence:
    def __init__(
        self,
        *,
        state: str = "healthy",
        ready_at: float = 1,
        body_status: str = "healthy",
    ) -> None:
        self.state = state
        self.ready_at = ready_at
        self.body_status = body_status
        self.now = 0.0
        self.sleeps: list[float] = []
        self.recreates: list[list[str]] = []
        self.listings: list[list[str]] = []
        self.effects_probes = 0

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def run(
        self, cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        stdout = ""
        returncode = 0
        if cmd[:2] == ["docker", "ps"] and any("label=" in arg for arg in cmd):
            self.listings.append(cmd)
            stdout = "effects-container\n"
        elif cmd[0] == "bash":
            function = cmd[5]
            stdout = {
                "runtime_health_budget_seconds": "2000",
                "runtime_started_at": "t0",
                "runtime_health_state": self.state,
                "runtime_container_status": "exited"
                if self.state == "not-running"
                else "running",
            }.get(function, "")
        elif "--force-recreate" in cmd:
            self.recreates.append(cmd)
            self.state = "starting"
        elif "omnidash_analytics" in cmd:
            stdout = "t"
        elif cmd[0] == "curl":
            assert "--fail" in cmd, "HTTP 503 must not look like a successful transport"
            is_effects = "http://localhost:8086/health" in cmd
            if is_effects:
                self.effects_probes += 1
            if is_effects and self.now < self.ready_at:
                returncode = 22
                stdout = "HTTP 503"
            else:
                stdout = json.dumps(
                    {
                        "status": self.body_status if is_effects else "healthy",
                        "details": {"is_running": True, "config_prefetch_status": "ok"},
                    }
                )
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch) -> _Convergence:
    lane = _Convergence()
    monkeypatch.setattr(executor_mod, "_run", lane.run)
    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", lane.sleep)
    monkeypatch.setattr(executor_mod.time, "monotonic", lambda: lane.now)
    return lane


def test_transient_503_docker_healthy_retries_without_recreate(
    replay: _Convergence,
) -> None:
    checks = DeployExecutor().verify(lambda *_args: None)
    assert replay.effects_probes >= 2
    assert replay.sleeps
    assert replay.sleeps[0] == 1
    assert replay.recreates == []
    assert all(check.status == "pass" for check in checks)


def test_healthy_docker_does_not_get_recreated_for_persistent_host_failure(
    replay: _Convergence,
) -> None:
    replay.ready_at = 10000
    with pytest.raises(VerificationFailedError):
        DeployExecutor().verify(lambda *_args: None)
    assert replay.effects_probes > 1
    assert replay.recreates == []
    assert replay.now <= 2000


@pytest.mark.parametrize("state", ["starting", "absent", "restarted", "no-healthcheck"])
def test_no_recreate_without_docker_failure(replay: _Convergence, state: str) -> None:
    replay.state = state
    replay.ready_at = 10000
    with pytest.raises(VerificationFailedError):
        DeployExecutor().verify(lambda *_args: None)
    assert replay.recreates == []


@pytest.mark.parametrize("state", ["unhealthy", "not-running"])
def test_slow_recreated_effects_uses_its_own_health_budget(
    replay: _Convergence, state: str
) -> None:
    replay.state = state
    replay.ready_at = 900
    executor = DeployExecutor()
    executor.verify(lambda *_args: None)
    assert len(replay.recreates) == 1
    assert replay.recreates[0][-1] == "runtime-effects"
    assert executor.verify_recreate[0].readiness_budget_seconds >= 1950
    assert any("--all" in command for command in replay.listings)
    assert replay.now >= 900


def test_slow_start_is_not_recreated_inside_start_period(replay: _Convergence) -> None:
    replay.state = "starting"
    replay.ready_at = 900
    DeployExecutor().verify(lambda *_args: None)
    assert replay.now >= 900
    assert replay.recreates == []


def test_docker_healthy_is_authoritative_for_answering_200(
    replay: _Convergence,
) -> None:
    replay.ready_at = 0
    replay.body_status = "unhealthy"
    checks = DeployExecutor().verify(lambda *_args: None)
    assert (
        next(check for check in checks if check.service == "runtime-effects").status
        == "pass"
    )
    assert replay.recreates == []


def test_effects_budget_is_read_from_compose_even_without_dependents(
    tmp_path: Path,
) -> None:
    from deploy_agent.compose_budget import derive_runtime_health_budget

    compose = tmp_path / "compose.yaml"
    compose.write_text(
        "services:\n  runtime-effects:\n    healthcheck:\n      start_period: 1800s\n      interval: 30s\n      timeout: 10s\n      retries: 5\n"
    )
    assert (
        derive_runtime_health_budget(
            (str(compose),), "runtime-effects", floor_seconds=600
        )
        >= 1950
    )


def test_reprobe_has_a_wall_clock_deadline(replay: _Convergence) -> None:
    replay.state = "unhealthy"
    replay.ready_at = 10000
    real_run = replay.run

    def slow_run(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "curl":
            replay.now += 9
        return real_run(cmd, timeout, **kwargs)

    with patch("deploy_agent.executor._run", side_effect=slow_run):
        with pytest.raises(VerificationFailedError):
            DeployExecutor().verify(lambda *_args: None)
    assert len(replay.recreates) == 1
    assert replay.now <= 2030


def test_failure_state_is_read_again_before_recreation(replay: _Convergence) -> None:
    replay.state = "unhealthy"
    replay.ready_at = 10000
    real_run = replay.run
    state_reads = 0

    def recovered_docker(
        cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal state_reads
        if cmd[0] == "bash" and cmd[5] == "runtime_health_state":
            state_reads += 1
            if state_reads == 2:
                replay.state = "healthy"
        return real_run(cmd, timeout, **kwargs)

    with patch("deploy_agent.executor._run", side_effect=recovered_docker):
        with pytest.raises(VerificationFailedError):
            DeployExecutor().verify(lambda *_args: None)
    assert state_reads == 2
    assert replay.recreates == []


@pytest.mark.parametrize(
    ("overlay_health", "budget"),
    [("      interval: 60s\n", 2150), ("      disable: true\n", 600)],
)
def test_health_budget_honours_overlay(
    tmp_path: Path, overlay_health: str, budget: int
) -> None:
    from deploy_agent.compose_budget import derive_runtime_health_budget

    base = tmp_path / "base.yaml"
    base.write_text(
        "services:\n  runtime-effects:\n    healthcheck:\n      start_period: 1800s\n      interval: 30s\n      timeout: 10s\n      retries: 5\n"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "services:\n  runtime-effects:\n    healthcheck:\n" + overlay_health
    )
    assert (
        derive_runtime_health_budget(
            (str(base), str(overlay)), "runtime-effects", floor_seconds=600
        )
        == budget
    )


def test_health_budget_honours_explicit_mapping_override(tmp_path: Path) -> None:
    from deploy_agent.compose_budget import derive_runtime_health_budget

    base = tmp_path / "base.yaml"
    base.write_text(
        "services:\n  runtime-effects:\n    healthcheck:\n      start_period: 1800s\n"
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "services:\n  runtime-effects:\n    healthcheck: !override\n      start_period: 30s\n"
    )
    assert (
        derive_runtime_health_budget(
            (str(base), str(overlay)), "runtime-effects", floor_seconds=0
        )
        == 210
    )


def test_unreadable_compose_budget_prevents_recreation(replay: _Convergence) -> None:
    from deploy_agent.compose_budget import ComposeBudgetError

    replay.state = "unhealthy"
    with patch(
        "deploy_agent.executor.runtime_health_probe_budget",
        side_effect=ComposeBudgetError("unreadable model"),
    ):
        with pytest.raises(ComposeBudgetError, match="unreadable model"):
            DeployExecutor().verify(lambda *_args: None)
    assert replay.recreates == []
