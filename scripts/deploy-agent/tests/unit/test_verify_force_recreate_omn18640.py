# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed post-deploy health probe recreates that one container (OMN-18640 AC7).

MEASURED on the ``.201`` dev lane, 2026-09-18T23:16Z onward. The deploy agent
recreated ``omnibase-infra-redpanda``; ``omninode-runtime-effects`` was not
recreated with it, lost its group coordinator, logged 21,200
``GroupCoordinatorNotAvailableError`` lines and never fetched again. Its
consumer groups stayed ``Stable`` with assigned partitions and frozen lag, so
every liveness surface read green while no OCC evidence companion was minted
anywhere in the fleet and every open PR held a red Receipt Gate.

The agent SAW it. Jobs ``a8d7acbb-d142-4d64-8c57-256682e5463c`` and
``f0789d4a`` both recorded ``runtime: success`` and then failed at
``executor.verify`` on this lane's effects port::

    Command '['curl', '-sS', '--max-time', '10',
    'http://localhost:8086/health']' timed out after 10 seconds

and then left the container exactly as it found it, because the remedy
available to the runtime phase is an ``up -d`` that is a no-op for a container
whose image and config hash have not changed. A container that is GONE gets
recreated; a container that is RUNNING AND USELESS does not. Autoheal cannot
cover it either -- ``docker/docker-compose.dev-lane.yml`` strips
``autoheal=true`` from the dev-lane runtime containers under the OMN-17562
operator ruling. So the job failed, the next job repeated the sequence, and the
lane could not recover itself for 97 minutes.

These tests drive ``DeployExecutor.verify`` through a fake compose runner.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import (
    SCOPE_SERVICES,
    EnumRuntimeLane,
    EnumVerifyRecreateOutcome,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import (
    VERIFY_RECREATE_LANES,
    DeployExecutor,
    lane_config_for,
)

# The compose projects this agent may never hand a recreate to. ``judge`` and
# ``lakshman`` are not even lanes in ``EnumRuntimeLane``, which is why they are
# named HERE as literal project strings: a future lane addition that made one of
# them reachable would have to pass this assertion, and nothing else in the
# package would notice.
GOVERNED_COMPOSE_PROJECTS = (
    "omnibase-infra-stability-test",
    "omnibase-infra-judge",
    "omnibase-infra-lakshman",
    "omnibase-infra-prod",
)


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _completed(
    cmd: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=cmd, returncode=returncode, stdout=stdout, stderr=stderr
    )


def _health_payload(*, status: str = "healthy", is_running: bool = True) -> str:
    return json.dumps(
        {
            "status": status,
            "details": {"is_running": is_running, "config_prefetch_status": "ok"},
        }
    )


def _is_projection_table_check(cmd: list[str]) -> bool:
    return "omnidash_analytics" in cmd and any(
        f"SELECT to_regclass('public.{table}') IS NOT NULL" in cmd
        for table in ("delegation_events", "node_service_registry")
    )


def _is_health_probe(cmd: list[str], port: int) -> bool:
    return f"http://localhost:{port}/health" in cmd


def _is_compose_recreate(cmd: list[str]) -> bool:
    return cmd[:2] == ["docker", "compose"] and "--force-recreate" in cmd


def _recreated_services(cmds: list[list[str]]) -> list[str]:
    """Return the trailing service name of every force-recreate argv issued."""
    return [cmd[-1] for cmd in cmds if _is_compose_recreate(cmd)]


class _Lane:
    """A fake compose runner over one lane's health ports.

    ``effects_health`` is a list of the outcomes the effects probe returns in
    order, the last one repeating: ``"timeout"`` raises
    ``subprocess.TimeoutExpired`` exactly as the live probe did, ``"unhealthy"``
    returns a parseable body that is not healthy, ``"healthy"`` passes.
    """

    def __init__(
        self,
        *,
        effects_health: list[str],
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
        recreate_returncode: int = 0,
    ) -> None:
        self.effects_health = effects_health
        self.lane = lane
        self.recreate_returncode = recreate_returncode
        self.cmds: list[list[str]] = []
        self._effects_probes = 0
        runtime_port, effects_port = (
            port for _service, port in lane_config_for(lane).runtime_health_targets
        )
        self.runtime_port = runtime_port
        self.effects_port = effects_port

    def __call__(
        self, cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess:
        self.cmds.append(list(cmd))
        if cmd[:2] == ["docker", "ps"]:
            return _completed(cmd)
        if _is_projection_table_check(cmd):
            return _completed(cmd, stdout="t\n")
        if _is_compose_recreate(cmd):
            return _completed(cmd, returncode=self.recreate_returncode)
        if _is_health_probe(cmd, self.runtime_port):
            return _completed(cmd, stdout=_health_payload())
        if _is_health_probe(cmd, self.effects_port):
            index = min(self._effects_probes, len(self.effects_health) - 1)
            self._effects_probes += 1
            outcome = self.effects_health[index]
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=10)
            if outcome == "unhealthy":
                return _completed(cmd, stdout=_health_payload(status="degraded"))
            return _completed(cmd, stdout=_health_payload())
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")


@pytest.fixture(autouse=True)
def _do_not_sleep_through_the_readiness_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the post-recreate readiness poll instant.

    A recreated runtime takes minutes to bind its health port -- the measured
    force-recreate on this lane was 336s with :8085 bound at t+321s -- so the
    executor waits on a budget derived from the compose model rather than
    re-probing into a container that is still booting. The WAIT is what this
    fixture removes; the poll loop, the budget derivation and the number of
    probes are all still real.
    """
    from deploy_agent import executor as executor_mod

    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", lambda _seconds: None)


def test_a_failed_effects_probe_force_recreates_exactly_that_service() -> None:
    """AC7: the failing service is recreated; the other runtime services are not."""
    executor = DeployExecutor()
    lane = _Lane(effects_health=["timeout", "healthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == ["runtime-effects"]
    recreate = next(cmd for cmd in lane.cmds if _is_compose_recreate(cmd))
    assert "--no-deps" in recreate
    assert "-p" in recreate
    assert recreate[recreate.index("-p") + 1] == "omnibase-infra"
    # The other two members of the dev lane's runtime family are never named.
    assert "omninode-runtime" not in _recreated_services(lane.cmds)
    assert "runtime-worker" not in _recreated_services(lane.cmds)

    status_by_service = {check.service: check.status for check in checks}
    assert status_by_service["runtime-effects"] == "pass"
    assert status_by_service["omninode-runtime"] == "pass"


def test_the_recreate_is_recorded_on_the_job_record_not_only_logged() -> None:
    executor = DeployExecutor()
    lane = _Lane(effects_health=["timeout", "healthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        executor.verify(on_phase_update=_noop_phase_update)

    assert len(executor.verify_recreate) == 1
    record = executor.verify_recreate[0]
    assert record.service == "runtime-effects"
    assert record.lane == EnumRuntimeLane.DEV
    assert record.compose_project == "omnibase-infra"
    assert record.endpoint == "http://localhost:8086/health"
    assert record.outcome == EnumVerifyRecreateOutcome.RECOVERED


def test_a_still_failing_probe_after_the_single_retry_still_fails_the_job() -> None:
    """One recreate, not a loop -- and the job still fails when it did not help."""
    executor = DeployExecutor()
    lane = _Lane(effects_health=["timeout"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(subprocess.TimeoutExpired):
            executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == ["runtime-effects"]
    assert executor.verify_recreate[0].outcome == (
        EnumVerifyRecreateOutcome.STILL_FAILING
    )


def test_an_unhealthy_body_is_a_verification_failure_too() -> None:
    """The probe that returns a parseable, not-healthy body recreates as well."""
    executor = DeployExecutor()
    lane = _Lane(effects_health=["unhealthy", "healthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == ["runtime-effects"]
    status_by_service = {check.service: check.status for check in checks}
    assert status_by_service["runtime-effects"] == "pass"


def test_a_healthy_lane_is_never_recreated() -> None:
    """The positive control: nothing failing, nothing recreated, field present."""
    executor = DeployExecutor()
    lane = _Lane(effects_health=["healthy"])

    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == []
    assert executor.verify_recreate == []
    assert all(check.status == "pass" for check in checks)


def test_a_failed_recreate_command_is_recorded_and_does_not_mask_the_failure() -> None:
    executor = DeployExecutor()
    lane = _Lane(effects_health=["timeout"], recreate_returncode=1)

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(subprocess.TimeoutExpired):
            executor.verify(on_phase_update=_noop_phase_update)

    assert executor.verify_recreate[0].outcome == (
        EnumVerifyRecreateOutcome.RECREATE_FAILED
    )
    assert executor.verify_recreate[0].recreate_returncode == 1


@pytest.mark.parametrize("lane", [EnumRuntimeLane.STABILITY_TEST, EnumRuntimeLane.PROD])
def test_a_non_dev_lane_never_takes_the_recreate_path(lane: EnumRuntimeLane) -> None:
    """The governed lanes keep today's behaviour exactly: fail, touch nothing.

    Two independent reasons, and either alone is sufficient. The lanes are
    governed surfaces whose containers this remedy is not authorised to bounce;
    and their ``runtime_health_targets`` carry CONTAINER names, not compose
    SERVICE names, so the argv this path builds would abort on `no such
    service` there anyway.
    """
    executor = DeployExecutor()
    fake = _Lane(effects_health=["timeout"], lane=lane)

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(subprocess.TimeoutExpired):
            executor.verify(on_phase_update=_noop_phase_update, lane=lane)

    assert _recreated_services(fake.cmds) == []
    assert executor.verify_recreate == []


def test_no_governed_compose_project_is_ever_named_on_any_lane() -> None:
    """Sweep every lane the agent knows and read the argv, not the intent."""
    for lane in EnumRuntimeLane:
        executor = DeployExecutor()
        fake = _Lane(effects_health=["timeout", "healthy"], lane=lane)
        with patch("deploy_agent.executor._run", side_effect=fake):
            try:
                executor.verify(on_phase_update=_noop_phase_update, lane=lane)
            except subprocess.TimeoutExpired:
                pass
        for cmd in fake.cmds:
            if not _is_compose_recreate(cmd):
                continue
            joined = " ".join(cmd)
            for project in GOVERNED_COMPOSE_PROJECTS:
                assert project not in joined, (
                    f"lane {lane.value} built a recreate naming {project}: {joined}"
                )


def test_only_the_dev_lane_is_in_the_recreate_allowlist() -> None:
    """Adding a lane here is a decision, so it is a red test rather than a diff."""
    assert frozenset({EnumRuntimeLane.DEV}) == VERIFY_RECREATE_LANES


def test_every_dev_health_target_is_a_compose_service_of_the_runtime_scope() -> None:
    """The name handed to ``docker compose up`` must be a service it resolves.

    Anti-drift, and the reason the path is dev-only: on this lane the health
    targets are compose service names, and on no other lane are they.
    """
    dev_targets = {
        service
        for service, _port in lane_config_for(
            EnumRuntimeLane.DEV
        ).runtime_health_targets
    }
    assert dev_targets <= set(SCOPE_SERVICES[Scope.RUNTIME])


def test_the_agent_threads_the_recreate_record_onto_the_terminal_event() -> None:
    """The executor's record has to REACH the event, not merely exist.

    Read from the source because the alternative is driving a whole job against
    a live broker; the field's own shape is asserted by the model tests above
    and by ``build_completion_payload``'s own test below.
    """
    agent_source = (
        Path(__file__).resolve().parents[2] / "deploy_agent" / "agent.py"
    ).read_text(encoding="utf-8")
    assert "verify_recreate=self.executor.verify_recreate" in agent_source


def test_the_terminal_event_carries_the_field_even_when_nothing_was_recreated() -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    from deploy_agent.job_state import JobState
    from deploy_agent.publisher import build_completion_payload

    job = JobState(
        correlation_id=uuid4(),
        command={"git_ref": "dev", "scope": "runtime", "runtime_lane": "dev"},
        phase_results={Phase.VERIFICATION: PhaseStatus.SUCCESS},
        completed_at=datetime.now(UTC),
    )
    payload = build_completion_payload(job, "0" * 40, [])
    assert payload["verify_recreate"] == []
