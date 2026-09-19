# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A verification that did not pass fails the job (OMN-18640 AC8).

WHAT WAS GREEN THAT SHOULD NOT HAVE BEEN. ``DeployExecutor.verify`` recorded a
``ModelHealthCheck`` with ``status="fail"`` for a probe that answered quickly
and unhealthily, and then called ``on_phase_update(VERIFICATION, SUCCESS)`` and
returned the list. Only a ``subprocess.TimeoutExpired`` failed the job. So a
runtime that answered ``{"status": "degraded"}`` in 12 ms, a runtime whose
``/health`` refused the connection outright, and a lane missing a required
projection table were each recorded as failures on the terminal event and
reported as a SUCCESSFUL deploy.

The 2026-09-18 wedge was caught only because the dead effects container stopped
answering at all and the probe hit its ten-second ceiling. A container that
answers "I am degraded" fast enough is the same outage with a green deploy over
it -- the identical class as every other green-through-outage surface found
that night.

WHAT IS DELIBERATELY NOT IN THE VERDICT, and this is measured rather than
assumed: the two ``docker ps --filter`` checks are HOST-WIDE. They name no
lane, no compose project and no service, so ``health=unhealthy`` matches any
container on the box. Read on the lab host at 2026-09-19T03:09Z, it returned
``omninode-pypi-cache`` -- an unrelated container, unhealthy right then, while
both dev-lane runtimes were fine. Putting that filter in the verdict would have
failed EVERY deploy on EVERY lane at that moment, including the dev lane's own
recovery deploy. It stays recorded and advisory until it is lane-scoped, which
is a separate change; ``test_a_host_wide_unhealthy_container_does_not_fail_the
_lane_deploy`` pins the carve-out so it is a decision and not an oversight.

These tests drive ``DeployExecutor.verify`` through a fake command runner.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    EnumRuntimeLane,
    EnumVerifyRecreateOutcome,
    Phase,
    PhaseStatus,
)
from deploy_agent.executor import (
    DeployExecutor,
    VerificationFailedError,
    lane_config_for,
)


class _Phases:
    """Records every phase verdict the executor reports, in order."""

    def __init__(self) -> None:
        self.updates: list[tuple[Phase, PhaseStatus]] = []

    def __call__(self, phase: Phase, status: PhaseStatus) -> None:
        self.updates.append((phase, status))

    def verdicts_for(self, phase: Phase) -> list[PhaseStatus]:
        return [status for recorded, status in self.updates if recorded == phase]


def _completed(
    cmd: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=cmd, returncode=returncode, stdout=stdout, stderr=stderr
    )


def _health_payload(
    *,
    status: str = "healthy",
    is_running: bool = True,
    config_prefetch_status: str = "ok",
) -> str:
    return json.dumps(
        {
            "status": status,
            "details": {
                "is_running": is_running,
                "config_prefetch_status": config_prefetch_status,
            },
        }
    )


def _is_projection_table_check(cmd: list[str]) -> bool:
    return "omnidash_analytics" in cmd and any(
        f"SELECT to_regclass('public.{table}') IS NOT NULL" in cmd
        for table in ("delegation_events", "node_service_registry")
    )


def _is_health_probe(cmd: list[str], port: int) -> bool:
    return f"http://localhost:{port}/health" in cmd


class _Lane:
    """A fake command runner over one lane's health ports.

    ``effects_health`` is the sequence of outcomes the effects probe returns,
    the last one repeating. ``"timeout"`` raises ``subprocess.TimeoutExpired``
    exactly as the live probe did on 2026-09-18; ``"unhealthy"`` answers FAST
    with a parseable, not-healthy body, which is the case this file exists for;
    ``"refused"`` answers fast with curl's own connection-refused exit; and
    ``"healthy"`` passes.
    """

    def __init__(
        self,
        *,
        effects_health: list[str],
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        projection_tables_present: bool = True,
        unhealthy_containers: str = "",
    ) -> None:
        self.effects_health = effects_health
        self.lane = lane
        self.projection_tables_present = projection_tables_present
        self.unhealthy_containers = unhealthy_containers
        self.cmds: list[list[str]] = []
        self._effects_probes = 0
        runtime_port, effects_port = (
            port for _service, port in lane_config_for(lane).runtime_health_targets
        )
        self.runtime_port = runtime_port
        self.effects_port = effects_port

    def __call__(
        self, cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.cmds.append(list(cmd))
        if cmd[:2] == ["docker", "ps"]:
            if "health=unhealthy" in cmd:
                return _completed(cmd, stdout=self.unhealthy_containers)
            return _completed(cmd)
        if _is_projection_table_check(cmd):
            return _completed(
                cmd, stdout="t\n" if self.projection_tables_present else "f\n"
            )
        if cmd[:2] == ["docker", "compose"]:
            return _completed(cmd)
        if _is_health_probe(cmd, self.runtime_port):
            return _completed(cmd, stdout=_health_payload())
        if _is_health_probe(cmd, self.effects_port):
            index = min(self._effects_probes, len(self.effects_health) - 1)
            self._effects_probes += 1
            outcome = self.effects_health[index]
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=10)
            if outcome == "unhealthy":
                return _completed(
                    cmd, stdout=_health_payload(config_prefetch_status="degraded_error")
                )
            if outcome == "refused":
                return _completed(
                    cmd,
                    returncode=7,
                    stderr="curl: (7) Failed to connect to localhost port 8086",
                )
            return _completed(cmd, stdout=_health_payload())
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")


@pytest.fixture(autouse=True)
def _do_not_sleep_through_the_readiness_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the post-recreate readiness WAIT, keeping the poll loop real."""
    from deploy_agent import executor as executor_mod

    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", lambda _seconds: None)


def test_a_fast_unhealthy_body_fails_the_job() -> None:
    """AC8, the headline. Answering unhealthy in milliseconds is not a success.

    RED before this change: ``verify`` returned the check list and reported
    ``VERIFICATION SUCCESS``, so the agent completed the job ``success`` with a
    ``fail`` health check sitting on its own terminal event.
    """
    executor = DeployExecutor()
    phases = _Phases()
    lane = _Lane(effects_health=["unhealthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=phases)

    assert PhaseStatus.SUCCESS not in phases.verdicts_for(Phase.VERIFICATION)


def test_the_refusal_names_the_probe_and_carries_the_probes_own_reason() -> None:
    """The message is what reaches ``errors`` on the terminal event."""
    executor = DeployExecutor()
    lane = _Lane(effects_health=["unhealthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError) as excinfo:
            executor.verify(on_phase_update=_Phases())

    message = str(excinfo.value)
    assert "http://localhost:8086/health" in message
    assert "config_prefetch_status" in message
    assert "degraded_error" in message
    # The probe that PASSED is not named as a failure.
    assert "http://localhost:8085/health" not in message


def test_a_connection_refused_probe_fails_the_job_with_curls_own_exit() -> None:
    """Unreachable-and-fast is the other half of "not a pass"."""
    executor = DeployExecutor()
    lane = _Lane(effects_health=["refused"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError) as excinfo:
            executor.verify(on_phase_update=_Phases())

    assert "7" in str(excinfo.value)
    assert "http://localhost:8086/health" in str(excinfo.value)


def test_a_missing_projection_table_fails_the_job() -> None:
    """The lane-scoped postgres check is in the verdict too.

    It execs into THIS lane's postgres container, so unlike the two docker
    filters it cannot fail because of some other lane. Read on the lab host at
    2026-09-19T03:09Z both required tables were present on the dev lane and on
    the stability-test lane, so this refusal is not armed against live state.
    """
    executor = DeployExecutor()
    lane = _Lane(effects_health=["healthy"], projection_tables_present=False)

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError) as excinfo:
            executor.verify(on_phase_update=_Phases())

    assert "delegation_events" in str(excinfo.value)


def test_a_host_wide_unhealthy_container_does_not_fail_the_lane_deploy() -> None:
    """The measured carve-out. ``docker ps --filter`` names no lane.

    On the lab host at 2026-09-19T03:09Z that filter returned
    ``omninode-pypi-cache`` while both dev-lane runtimes were healthy. In the
    verdict it would have failed every deploy on every lane, the dev lane's own
    recovery deploy included. Recorded, not verdict-bearing, until it is
    lane-scoped.
    """
    executor = DeployExecutor()
    phases = _Phases()
    lane = _Lane(effects_health=["healthy"], unhealthy_containers="omninode-pypi-cache")

    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=phases)

    assert PhaseStatus.SUCCESS in phases.verdicts_for(Phase.VERIFICATION)
    docker_checks = {
        check.endpoint: check.status for check in checks if check.service == "docker"
    }
    assert docker_checks["docker ps --filter health=unhealthy"] == "fail"


def test_a_probe_the_recreate_repaired_still_passes() -> None:
    """Positive control: the AC7 remedy working is still a successful deploy."""
    executor = DeployExecutor()
    phases = _Phases()
    lane = _Lane(effects_health=["unhealthy", "healthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=phases)

    assert PhaseStatus.SUCCESS in phases.verdicts_for(Phase.VERIFICATION)
    assert all(check.status == "pass" for check in checks if check.service != "docker")
    assert executor.verify_recreate[0].outcome == EnumVerifyRecreateOutcome.RECOVERED


def test_a_timeout_still_raises_the_timeout_itself_unchanged() -> None:
    """#3802 is preserved exactly: the timeout object, not the new error.

    Its message is the string the agent already logs and the one both 2026-09-18
    job records carry, so a reader comparing occurrences is not handed a new
    shape for the case that already had one.
    """
    executor = DeployExecutor()
    lane = _Lane(effects_health=["timeout"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            executor.verify(on_phase_update=_Phases())

    assert not isinstance(excinfo.value, VerificationFailedError)
    assert (
        executor.verify_recreate[0].outcome == EnumVerifyRecreateOutcome.STILL_FAILING
    )


def test_the_health_checks_survive_the_refusal_for_the_terminal_event() -> None:
    """A job that fails verification must still publish what it probed.

    The agent's local ``health_checks`` is ``[]`` on this path -- the assignment
    is the statement that raised -- so before this change the terminal event of
    a verification failure carried NO health checks at all, on precisely the
    deploy whose probe readings mattered most.
    """
    executor = DeployExecutor()
    lane = _Lane(effects_health=["unhealthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_Phases())

    by_service = {check.service: check for check in executor.health_checks}
    assert by_service["runtime-effects"].status == "fail"
    assert "degraded_error" in by_service["runtime-effects"].detail
    assert by_service["omninode-runtime"].status == "pass"
    assert by_service["omninode-runtime"].detail == ""


def test_the_agent_publishes_the_executors_checks_when_its_own_are_empty() -> None:
    """Read from the source: the executor's record has to REACH the event."""
    from pathlib import Path

    agent_source = (
        Path(__file__).resolve().parents[2] / "deploy_agent" / "agent.py"
    ).read_text(encoding="utf-8")
    assert "health_checks or self.executor.health_checks" in agent_source


def test_the_four_verify_recreate_outcomes_are_unchanged() -> None:
    """Anti-drift on #3802: this change adds a verdict, not an outcome."""
    assert {outcome.value for outcome in EnumVerifyRecreateOutcome} == {
        "recovered",
        "still_failing",
        "recreate_failed",
        "recreate_timed_out",
    }


@pytest.mark.parametrize("lane", [EnumRuntimeLane.STABILITY_TEST, EnumRuntimeLane.PROD])
def test_the_verdict_holds_on_every_lane_without_recreating_anything(
    lane: EnumRuntimeLane,
) -> None:
    """A governed lane gets the honest verdict and no remedy.

    The AC7 recreate is dev-only and stays dev-only. Failing closed is not a
    mutation, so it applies everywhere: a governed lane reporting an unhealthy
    runtime has not had a successful deploy either.
    """
    executor = DeployExecutor()
    fake = _Lane(effects_health=["unhealthy"], lane=lane)

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_Phases(), lane=lane)

    assert executor.verify_recreate == []
    assert not any(cmd[:2] == ["docker", "compose"] for cmd in fake.cmds)
