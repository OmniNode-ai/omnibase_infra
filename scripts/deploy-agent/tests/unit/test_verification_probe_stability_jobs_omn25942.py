# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Post-deploy verification probes one lane and waits for its runtime (OMN-25942).

Replays the three stability lane jobs of 2026-11-19 (fixture
``fixtures/omn25942/stability_jobs_verification_probe.json``): 1f7f39c6, 6ff2c3f5 and
914b6b79 each failed in verification with ``curl exited 56 ... Connection reset by
peer`` on ``:18085/health``, because the recreated runtime was still inside its
declared start and the probe read a failure at once. The wait that fixed it is
omnibase_infra#6013; the worker that never booted is omnibase_infra#6023.

Two properties are pinned here, each by the recorded shape:

* the lane's verification reads only that lane's own services, ports and postgres;
* a recreated runtime that resets the connection while Docker reports it starting is
  waited for, and only a runtime that never answers inside its budget is read as a
  failure.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import (
    VERIFY_RECREATE_LANES,
    DeployExecutor,
    VerificationFailedError,
    lane_config_for,
)

from tests.unit import test_verify_stability_start_wait_omn24562 as wait_tests

pytestmark = pytest.mark.unit

fake_bin = wait_tests.fake_bin
_real_helpers_no_sleep = wait_tests._real_helpers_no_sleep

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "omn25942"
    / "stability_jobs_verification_probe.json"
)
RECORDED = json.loads(FIXTURE.read_text())

STABILITY = EnumRuntimeLane.STABILITY_TEST
# The lane's own configuration names the runtime container; the record is read with it.
RUNTIME_SERVICE = lane_config_for(STABILITY).runtime_health_targets[0][0]
RECORDED_ERRORS = [
    job["error_template"].format(runtime_service=RUNTIME_SERVICE)
    for job in RECORDED["jobs"]
]
CONNECTION_RESET = "curl: (56) Recv failure: Connection reset by peer"
LANE_PORTS = {
    lane: {port for _, port in lane_config_for(lane).runtime_health_targets}
    for lane in EnumRuntimeLane
}


class _ResetWhileStarting(wait_tests._StartingLane):
    """The recorded runtime: the port accepts, then resets, until it is ready."""

    def __call__(
        self, cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = super().__call__(cmd, timeout, **kwargs)
        if result.returncode == 7 and "curl" in cmd:
            return wait_tests._completed(cmd, returncode=56, stderr=CONNECTION_RESET)
        return result


def test_verification_probe_recorded_jobs_are_one_failure_on_the_runtime_port() -> None:
    assert len(RECORDED["jobs"]) == 3
    assert {job["runtime_lane"] for job in RECORDED["jobs"]} == {STABILITY.name}
    assert {job["status"] for job in RECORDED["jobs"]} == {"failed"}
    for error in RECORDED_ERRORS:
        assert error.startswith("post-deploy verification failed: ")
        assert "http://localhost:18085/health" in error
        assert "curl exited 56" in error
        assert CONNECTION_RESET in error
    readings = RECORDED["container_readings_utc_2026_11_19"]
    assert readings["runtime"]["published_ports"] == [
        18085,
        18086,
    ]
    assert readings["pre_fold_runtime_effects_container_present"] is False


def test_verification_probe_without_the_start_wait_reads_the_recorded_failure(
    fake_bin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: with no wait the same replay fails with the recorded error text."""
    monkeypatch.setattr(executor_mod, "VERIFY_START_WAIT_LANES", VERIFY_RECREATE_LANES)
    fake = _ResetWhileStarting(
        lane=STABILITY,
        fake_bin=fake_bin,
        runtime_fails=3,
        docker_env=wait_tests._starting(),
    )

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError) as raised:
            DeployExecutor().verify(
                on_phase_update=wait_tests._noop_phase_update, lane=STABILITY
            )

    assert str(raised.value) == RECORDED_ERRORS[0]
    assert fake.runtime_probes == 1


def test_verification_probe_waits_for_the_recreated_runtime_before_reading_a_failure(
    fake_bin: Path,
) -> None:
    """The fix: the same resets are waited out and the runtime then passes."""
    fake = _ResetWhileStarting(
        lane=STABILITY,
        fake_bin=fake_bin,
        runtime_fails=3,
        docker_env=wait_tests._starting(),
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=fake):
        checks = executor.verify(
            on_phase_update=wait_tests._noop_phase_update, lane=STABILITY
        )

    runtime = next(c for c in checks if c.endpoint.endswith(":18085/health"))
    assert runtime.status == "pass"
    assert fake.runtime_probes == 4
    assert fake.recreates() == []
    assert executor.verify_recreate == []


def test_verification_probe_reads_a_failure_once_the_declared_start_is_spent(
    fake_bin: Path,
) -> None:
    """A runtime that never answers is still a failure, after its own budget."""
    fake = _ResetWhileStarting(
        lane=STABILITY,
        fake_bin=fake_bin,
        runtime_fails=None,
        docker_env=wait_tests._starting(FAKE_HEALTHCHECK=wait_tests.SHORT_HEALTHCHECK),
    )

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError) as raised:
            DeployExecutor().verify(
                on_phase_update=wait_tests._noop_phase_update, lane=STABILITY
            )

    assert CONNECTION_RESET in str(raised.value)
    assert fake.runtime_probes > 1
    assert fake.recreates() == []


@pytest.mark.parametrize("lane", [STABILITY, EnumRuntimeLane.DEV])
def test_verification_probe_reads_only_its_own_lane(
    fake_bin: Path, lane: EnumRuntimeLane
) -> None:
    """Every probe of one lane's verification names that lane's ports and postgres."""
    fake = wait_tests._StartingLane(
        lane=lane,
        fake_bin=fake_bin,
        runtime_fails=2,
        docker_env=wait_tests._starting(),
    )

    with patch("deploy_agent.executor._run", side_effect=fake):
        DeployExecutor().verify(
            on_phase_update=wait_tests._noop_phase_update, lane=lane
        )

    other_ports = set().union(
        *(ports for other, ports in LANE_PORTS.items() if other is not lane)
    )
    probed = {
        int(match.group(1))
        for cmd in fake.cmds
        for part in cmd
        if (match := re.fullmatch(r"http://localhost:(\d+)/health", part))
    }
    assert probed == LANE_PORTS[lane]
    assert probed.isdisjoint(other_ports)
    own_postgres = lane_config_for(lane).postgres_container
    for other in EnumRuntimeLane:
        if other is lane:
            continue
        foreign = lane_config_for(other).postgres_container
        assert foreign == own_postgres or not any(foreign in cmd for cmd in fake.cmds)
