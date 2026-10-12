# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The 2026-10-11 lab dev-lane outage, and the three rules that close it.

OMN-20864. MEASURED on the lab dev lane (deploy-agent-dev journal, the dev broker's
``onex.evt.omnibase-infra.runtime-health-check.v1`` topic, ``docker ps``):

* 13:17Z job 72e74ec6 started the new ``omninode-runtime``. Its health monitor
  named ``projection_attachment`` on its first cycle (13:23:17Z) and
  ``projection_dlq_saturation`` (branch_claim_check_effect) on every cycle from
  13:28:21Z -- the same finding the container it replaced had carried since at
  least 12:41Z. ``/health`` answered ``status=degraded is_running=True``.
* 13:33:44Z verification probed it, waited 976 s while docker said
  ``starting``, then at 13:50:00Z (docker ``unhealthy``) force-recreated it.
  The recreate hit its 300 s ceiling and left ``d41bf2abebd1_omninode-runtime``
  Created with nothing running under the service name.
* 14:09-14:14Z job 8d863e24 timed out the same way on ``forward-migration``,
  leaving ``3cf1f627bfc8_omnibase-infra-forward-migration`` Created.
* From 14:14Z the idle branch read no running revision and refused
  (``running_unreadable``) every minute until a hand refresh at 19:41Z.

Rule 1: a runtime that answers running with its monitor degraded is not
recreated (the verdict still fails), and two consecutive monitor cycles on the
same dimensions end the start wait early. Rule 2: a recreate killed at its
ceiling has its Created leftover removed and is retried once. Rule 3: an idle
lane whose main runtime is absent or stopped is restored.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent.agent import DeployAgent
from deploy_agent.events import EnumRuntimeLane, EnumVerifyRecreateOutcome, Phase
from deploy_agent.executor import (
    DeployExecutor,
    ModelAnsweringDegraded,
    VerificationFailedError,
    _answering_degraded,
    lane_config_for,
    monitor_settled_degraded,
    remove_recreate_leftovers,
    run_compose_recreate,
)
from deploy_agent.idle_converge import (
    IDLE_RESTORE_REQUESTER,
    RESTORE_AFTER,
    RESTORE_RETRY_AFTER,
    EnumIdleRestoreVerdict,
    EnumRuntimePresence,
    ModelIdleRestoreInputs,
    decide_restore,
    read_runtime_presence,
)
from deploy_agent.job_state import JobStore
from deploy_agent.routing import ModelLaneFlags, ModelLaneFreeze

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPERS = REPO_ROOT / "scripts" / "runtime_build" / "runtime_health_wait.sh"
DEV = lane_config_for(EnumRuntimeLane.DEV)
RUNTIME_PORT, EFFECTS_PORT = (port for _svc, port in DEV.runtime_health_targets)
RUNTIME_CONTAINER_ID = "a" * 64
# The dev-lane runtime's declared healthcheck: 1800 s start, 30 s x 5, 10 s.
DECLARED = "1800000000000 30000000000 5 10000000000"

HEALTHY_BODY = json.dumps(
    {
        "status": "healthy",
        "details": {"is_running": True, "config_prefetch_status": "skipped"},
    }
)


def _degraded_body(observed_at: str, *dimensions: str) -> str:
    """The shape job 72e74ec6's runtime answered, read from the live endpoint."""
    return json.dumps(
        {
            "status": "degraded",
            "details": {
                "healthy": False,
                "degraded": True,
                "is_running": True,
                "startup_in_progress": False,
                "event_bus_healthy": True,
                "config_prefetch_status": "skipped",
                "runtime_health": {
                    "status": "DEGRADED",
                    "observed_at": observed_at,
                    "dimensions": [
                        {"name": "discovery_errors", "status": "HEALTHY"},
                        *({"name": d, "status": "DEGRADED"} for d in dimensions),
                    ],
                },
            },
        }
    )


def _completed(
    cmd: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _phase(_phase: Phase, _status: object) -> None:
    return None


# --- rule 1: what counts as "answering, monitor degraded" ---------------------


def _curl(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return _completed(["curl"], returncode=returncode, stdout=stdout)


def test_the_incident_body_is_answering_degraded() -> None:
    reading = _answering_degraded(
        _curl(_degraded_body("2026-10-11T13:33:23Z", "projection_dlq_saturation"))
    )
    assert reading == ModelAnsweringDegraded(
        observed_at="2026-10-11T13:33:23Z",
        dimensions=frozenset({"projection_dlq_saturation"}),
    )


@pytest.mark.parametrize(
    ("result", "why"),
    [
        (_curl("", returncode=7), "the port refused (the OMN-18640 wedge)"),
        (_curl("not json"), "not a health document"),
        (_curl(HEALTHY_BODY), "healthy"),
        (
            _curl(json.dumps({"status": "unhealthy", "details": {"is_running": True}})),
            "unhealthy is not degraded",
        ),
        (
            _curl(
                _degraded_body("t", "x").replace(
                    '"is_running": true', '"is_running": false'
                )
            ),
            "not running",
        ),
        (
            _curl(
                _degraded_body("t", "x").replace(
                    '"startup_in_progress": false', '"startup_in_progress": true'
                )
            ),
            "still starting",
        ),
        (
            _curl(
                _degraded_body("t", "x").replace(
                    '"event_bus_healthy": true', '"event_bus_healthy": false'
                )
            ),
            "its event bus is down, which a recreate can fix",
        ),
        (_curl(_degraded_body("t")), "the monitor names no dimension"),
    ],
)
def test_everything_else_is_not_answering_degraded(
    result: subprocess.CompletedProcess[str], why: str
) -> None:
    assert _answering_degraded(result) is None, why


def test_settled_means_two_cycles_on_the_same_dimensions() -> None:
    def r(at: str, *dims: str) -> ModelAnsweringDegraded:
        return ModelAnsweringDegraded(observed_at=at, dimensions=frozenset(dims))

    dlq = "projection_dlq_saturation"
    assert monitor_settled_degraded(r("13:28", dlq), r("13:33", dlq))
    assert not monitor_settled_degraded(r("13:28", dlq), r("13:28", dlq))
    assert not monitor_settled_degraded(
        r("13:23", "projection_attachment"), r("13:28", dlq)
    )
    assert not monitor_settled_degraded(None, r("13:33", dlq))
    assert not monitor_settled_degraded(r("", dlq), r("13:33", dlq))


# --- rule 1 in verify, against the real helpers and a fake docker -------------

_FAKE_DOCKER = """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == "inspect" ]] || { echo "unsupported: $1" >&2; exit 2; }
fmt="$3"
state="$(cat "${FAKE_STATE_FILE}")"
case "${fmt}" in
    *'printf "%d %d %d %d" .Config.Healthcheck.StartPeriod'*) printf '%s\\n' "${FAKE_HEALTHCHECK}" ;;
    *State.Status*) printf '%s t0\\n' "${state}" ;;
    *State.StartedAt*) printf 't0\\n' ;;
    *) echo "unsupported format: ${fmt}" >&2; exit 2 ;;
esac
"""


class _Lane:
    """The dev lane: the runtime answers ``bodies`` in order, then the last one.

    ``docker_states`` is what docker reports for the runtime on each helper
    read, in order, then the last one. Every compose command is recorded;
    ``recreate_hangs`` makes each ``--force-recreate`` raise its timeout.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        bodies: list[str],
        docker_states: list[str],
        recreate_hangs: int = 0,
        leftovers: str = "",
    ) -> None:
        self.bodies = bodies
        self.docker_states = docker_states
        self.recreate_hangs = recreate_hangs
        self.leftovers = leftovers
        self.cmds: list[list[str]] = []
        self.runtime_probes = 0
        self.state_reads = 0
        self.fake_bin = tmp_path / "bin"
        self.fake_bin.mkdir()
        docker = self.fake_bin / "docker"
        docker.write_text(_FAKE_DOCKER, encoding="utf-8")
        docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
        self.state_file = tmp_path / "state"

    def __call__(
        self, cmd: list[str], timeout: int, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.cmds.append(list(cmd))
        if cmd[0] == "bash":
            if "runtime_health_state" in cmd:
                state = self.docker_states[
                    min(self.state_reads, len(self.docker_states) - 1)
                ]
                self.state_reads += 1
            else:
                state = self.docker_states[0]
            self.state_file.write_text(state, encoding="utf-8")
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env={
                    **os.environ,
                    "PATH": f"{self.fake_bin}{os.pathsep}{os.environ['PATH']}",
                    "FAKE_HEALTHCHECK": DECLARED,
                    "FAKE_STATE_FILE": str(self.state_file),
                },
            )
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _completed(cmd, stdout=self.leftovers)
        if cmd[:2] == ["docker", "ps"]:
            if "label=com.docker.compose.service=omninode-runtime" in cmd:
                return _completed(cmd, stdout=f"{RUNTIME_CONTAINER_ID}\n")
            return _completed(cmd)
        if cmd[:3] == ["docker", "rm", "-f"]:
            return _completed(cmd)
        if "omnidash_analytics" in cmd:
            return _completed(cmd, stdout="t\n")
        if cmd[:2] == ["docker", "compose"] and "--force-recreate" in cmd:
            if self.recreate_hangs > 0:
                self.recreate_hangs -= 1
                raise subprocess.TimeoutExpired(cmd, timeout)
            return _completed(cmd)
        if f"http://localhost:{RUNTIME_PORT}/health" in cmd:
            body = self.bodies[min(self.runtime_probes, len(self.bodies) - 1)]
            self.runtime_probes += 1
            if body == "refused":
                return _completed(cmd, returncode=7, stderr="curl: (7) refused")
            return _completed(cmd, stdout=body)
        if f"http://localhost:{EFFECTS_PORT}/health" in cmd:
            return _completed(cmd, stdout=HEALTHY_BODY)
        return _completed(cmd, returncode=1, stderr=f"unexpected: {cmd}")

    @property
    def recreates(self) -> list[list[str]]:
        return [
            c
            for c in self.cmds
            if c[:2] == ["docker", "compose"] and "--force-recreate" in c
        ]


@pytest.fixture(autouse=True)
def _real_helpers_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    from deploy_agent import executor as executor_mod

    monkeypatch.setattr(executor_mod, "RUNTIME_HEALTH_WAIT_HELPERS", HELPERS)
    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", lambda _s: None)


def test_replay_72e74ec6_the_degraded_runtime_is_not_recreated(tmp_path: Path) -> None:
    """The measured sequence, end to end through verify.

    Probed at 13:33:44Z on the 13:33:23Z cycle (dlq), docker says starting;
    the next distinct cycle (13:38:28Z) names the same dimension, so the wait
    ends there -- not at 13:50Z -- and the runtime is left running.
    """
    dlq = "projection_dlq_saturation"
    lane = _Lane(
        tmp_path,
        bodies=[
            _degraded_body("2026-10-11T13:33:23Z", dlq),
            *[_degraded_body("2026-10-11T13:33:23Z", dlq)] * 29,
            _degraded_body("2026-10-11T13:38:28Z", dlq),
        ],
        docker_states=["running starting"],
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError) as raised:
            executor.verify(on_phase_update=_phase)

    assert lane.recreates == []
    assert [r.outcome for r in executor.verify_recreate] == [
        EnumVerifyRecreateOutcome.NOT_RECREATED_ANSWERING
    ]
    assert dlq in str(raised.value)
    assert "not recreated" in str(raised.value)
    # 1 up-front probe + 30 inside the wait, far short of the 2000 s budget's 200.
    assert lane.runtime_probes == 31


def test_a_degraded_runtime_docker_calls_unhealthy_is_not_recreated(
    tmp_path: Path,
) -> None:
    """13:50:00Z: the start period over, docker unhealthy, still answering."""
    lane = _Lane(
        tmp_path,
        bodies=[_degraded_body("2026-10-11T13:48:40Z", "projection_dlq_saturation")],
        docker_states=["running unhealthy"],
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_phase)

    assert lane.recreates == []
    assert executor.verify_recreate[0].outcome is (
        EnumVerifyRecreateOutcome.NOT_RECREATED_ANSWERING
    )


def test_a_boot_transient_degraded_cycle_is_waited_out_and_passes(
    tmp_path: Path,
) -> None:
    """19:36:15Z degraded on the first cycle, 19:41:22Z healthy: not a failure."""
    lane = _Lane(
        tmp_path,
        bodies=[
            "refused",
            _degraded_body(
                "2026-10-11T19:36:15Z",
                "projection_attachment",
                "projection_apply_divergence",
            ),
            _degraded_body(
                "2026-10-11T19:36:15Z",
                "projection_attachment",
                "projection_apply_divergence",
            ),
            HEALTHY_BODY,
        ],
        docker_states=["running starting"],
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=_phase)

    assert lane.recreates == []
    assert executor.verify_recreate == []
    assert next(c for c in checks if c.service == "omninode-runtime").status == "pass"


def test_positive_control_a_dead_port_is_still_recreated(tmp_path: Path) -> None:
    lane = _Lane(
        tmp_path,
        bodies=["refused", HEALTHY_BODY],
        docker_states=["running unhealthy"],
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        executor.verify(on_phase_update=_phase)

    assert len(lane.recreates) == 1
    assert executor.verify_recreate[0].outcome is EnumVerifyRecreateOutcome.RECOVERED


# --- rule 2: a timed-out recreate leaves nothing Created ----------------------

LEFTOVERS = (
    "d41bf2abebd1\td41bf2abebd1_omninode-runtime\tomninode-runtime\n"
    "3cf1f627bfc8\t3cf1f627bfc8_omnibase-infra-forward-migration\tforward-migration\n"
    "0123456789ab\tomnibase-infra-intelligence-migration\tintelligence-migration\n"
)


def test_leftover_removal_is_scoped_to_the_lane_the_shape_and_the_service() -> None:
    calls: list[list[str]] = []

    def run(
        cmd: list[str], timeout: int, **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return _completed(cmd, stdout=LEFTOVERS if cmd[1] == "ps" else "")

    with patch("deploy_agent.executor._run", side_effect=run):
        removed = remove_recreate_leftovers(EnumRuntimeLane.DEV, ["omninode-runtime"])

    assert removed == ["d41bf2abebd1_omninode-runtime"]
    listing = calls[0]
    assert f"label=com.docker.compose.project={DEV.compose_project}" in listing
    assert "status=created" in listing
    # A Created container under a plain name is not a recreate leftover.
    assert ["docker", "rm", "-f", "d41bf2abebd1"] in calls
    assert not [c for c in calls if c[:2] == ["docker", "rm"] and "0123456789ab" in c]


def test_a_recreate_that_times_out_is_cleaned_and_retried_once() -> None:
    attempts: list[str] = []

    def run(
        cmd: list[str], timeout: int, **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "compose"]:
            attempts.append("up")
            if len(attempts) == 1:
                raise subprocess.TimeoutExpired(cmd, timeout)
            return _completed(cmd)
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _completed(cmd, stdout=LEFTOVERS)
        return _completed(cmd)

    removed: list[str] = []
    with patch("deploy_agent.executor._run", side_effect=run):
        result = run_compose_recreate(
            ["docker", "compose", "up", "-d", "--force-recreate", "forward-migration"],
            timeout=300,
            lane=EnumRuntimeLane.DEV,
            services=["forward-migration"],
            removed=removed,
        )

    assert result.returncode == 0
    assert attempts == ["up", "up"]
    assert removed == ["3cf1f627bfc8_omnibase-infra-forward-migration"]


def test_a_second_timeout_is_cleaned_and_raised() -> None:
    def run(
        cmd: list[str], timeout: int, **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "compose"]:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _completed(cmd, stdout=LEFTOVERS)
        return _completed(cmd)

    removed: list[str] = []
    with patch("deploy_agent.executor._run", side_effect=run):
        with pytest.raises(subprocess.TimeoutExpired):
            run_compose_recreate(
                [
                    "docker",
                    "compose",
                    "up",
                    "-d",
                    "--force-recreate",
                    "omninode-runtime",
                ],
                timeout=300,
                lane=EnumRuntimeLane.DEV,
                services=["omninode-runtime"],
                removed=removed,
            )

    assert removed == ["d41bf2abebd1_omninode-runtime"] * 2


def test_replay_13_50z_the_verify_recreate_timeout_leaves_nothing_created(
    tmp_path: Path,
) -> None:
    """A dead port, a recreate that hangs once: cleaned, retried, recovered."""
    lane = _Lane(
        tmp_path,
        bodies=["refused", HEALTHY_BODY],
        docker_states=["running unhealthy"],
        recreate_hangs=1,
        leftovers=LEFTOVERS,
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        executor.verify(on_phase_update=_phase)

    assert len(lane.recreates) == 2
    assert ["docker", "rm", "-f", "d41bf2abebd1"] in lane.cmds
    assert executor.verify_recreate[0].outcome is EnumVerifyRecreateOutcome.RECOVERED


def test_replay_14_14z_the_migration_recreate_timeout_is_cleaned_and_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deploy_agent import executor as executor_mod

    monkeypatch.setattr(executor_mod, "DEV_LANE_ONLY_MIGRATION_SERVICES", ())
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", lambda *_a, **_k: (True, [])
    )
    hung: list[str] = []

    def run(
        cmd: list[str], timeout: int, **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "compose"] and "forward-migration" in cmd:
            if not hung:
                hung.append("forward-migration")
                raise subprocess.TimeoutExpired(cmd, timeout)
            return _completed(cmd)
        if cmd[:3] == ["docker", "ps", "-a"]:
            return _completed(cmd, stdout=LEFTOVERS)
        if "omnidash_analytics" in cmd:
            return _completed(cmd, stdout="t\n")
        return _completed(cmd)

    calls: list[list[str]] = []

    def recording(
        cmd: list[str], timeout: int, **kw: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return run(cmd, timeout, **kw)

    with patch("deploy_agent.executor._run", side_effect=recording):
        DeployExecutor()._ensure_runtime_migrations_ready(lane=EnumRuntimeLane.DEV)

    assert ["docker", "rm", "-f", "3cf1f627bfc8"] in calls
    assert (
        len([c for c in calls if "forward-migration" in c and c[1] == "compose"]) == 2
    )


# --- rule 3: a lane with no running runtime is restored -----------------------

NOW = datetime(2026, 10, 11, 14, 30, tzinfo=UTC)


def _restore_inputs(**overrides: Any) -> ModelIdleRestoreInputs:
    values: dict[str, Any] = {
        "now": NOW,
        "presence": EnumRuntimePresence.DOWN,
        "presence_detail": "no container named omninode-runtime",
        "job_active": False,
        "last_activity": datetime(2026, 10, 11, 14, 14, tzinfo=UTC),
        "last_attempt": None,
    }
    values.update(overrides)
    return ModelIdleRestoreInputs(**values)


@pytest.mark.parametrize(
    ("overrides", "verdict"),
    [
        ({}, EnumIdleRestoreVerdict.RESTORE),
        ({"presence": EnumRuntimePresence.RUNNING}, EnumIdleRestoreVerdict.NOT_DOWN),
        (
            {"presence": EnumRuntimePresence.UNKNOWN},
            EnumIdleRestoreVerdict.PRESENCE_UNREADABLE,
        ),
        ({"frozen_reason": "FROZEN until x"}, EnumIdleRestoreVerdict.FROZEN),
        ({"job_active": True}, EnumIdleRestoreVerdict.JOB_ACTIVE),
        (
            {"last_activity": NOW - RESTORE_AFTER + timedelta(seconds=1)},
            EnumIdleRestoreVerdict.RECENTLY_ACTIVE,
        ),
        (
            {"last_attempt": NOW - RESTORE_RETRY_AFTER + timedelta(seconds=1)},
            EnumIdleRestoreVerdict.RECENTLY_ATTEMPTED,
        ),
        (
            {"last_attempt": NOW - RESTORE_RETRY_AFTER},
            EnumIdleRestoreVerdict.RESTORE,
        ),
    ],
)
def test_decide_restore(
    overrides: dict[str, Any], verdict: EnumIdleRestoreVerdict
) -> None:
    assert decide_restore(_restore_inputs(**overrides)).verdict is verdict


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "presence"),
    [
        (0, "running\n", "", EnumRuntimePresence.RUNNING),
        (0, "restarting\n", "", EnumRuntimePresence.RUNNING),
        (0, "created\n", "", EnumRuntimePresence.DOWN),
        (0, "exited\n", "", EnumRuntimePresence.DOWN),
        (1, "", "Error: No such object: omninode-runtime", EnumRuntimePresence.DOWN),
        (1, "", "Cannot connect to the Docker daemon", EnumRuntimePresence.UNKNOWN),
    ],
)
def test_read_runtime_presence(
    returncode: int, stdout: str, stderr: str, presence: EnumRuntimePresence
) -> None:
    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return _completed(cmd, returncode=returncode, stdout=stdout, stderr=stderr)

    assert read_runtime_presence("omninode-runtime", run=run)[0] is presence


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.container_residue: list[object] = []
        self.sibling_source_refs: dict[str, str] = {}
        self.recreate_supervision: list[object] = []
        self.verify_recreate: list[object] = []
        self.deps_convergence: list[object] = []
        self.compose_invocations: list[object] = []
        self.health_checks: list[object] = []

    def __getattr__(self, name: str) -> Any:
        def record(*_args: object, **_kwargs: object) -> Any:
            self.calls.append(name)
            if name == "git_pull":
                return "d" * 40
            if name in ("rebuild_scope", "verify"):
                return []
            return None

        return record


def _agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, presence: EnumRuntimePresence
) -> DeployAgent:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    monkeypatch.setattr(agent_mod, "LAB_OVERLAY_ENABLED", False)
    agent = DeployAgent(skip_self_update=True)
    agent.job_store = JobStore(tmp_path / "jobs")
    monkeypatch.setattr(agent, "executor", _FakeExecutor())
    agent._idle_converge_started_at = NOW - timedelta(hours=1)
    agent._idle_converge_now = lambda: NOW
    agent._idle_read_runtime_presence = lambda: (
        presence,
        f"omninode-runtime {presence}",
    )

    def no_running_ref(repo: str) -> str | None:
        raise AssertionError("the converge rule read a revision on a down lane")

    agent._idle_read_running_ref = no_running_ref
    agent._idle_read_head_ref = lambda repo: "b" * 40
    agent._idle_probe_blocker = lambda now: None
    agent._lane_flags = ModelLaneFlags()
    return agent


def _jobs(tmp_path: Path) -> list[Any]:
    return [
        JobStore(tmp_path / "jobs").load(UUID(p.stem))
        for p in (tmp_path / "jobs").glob("*.json")
    ]


def test_replay_14_14z_a_down_lane_is_restored_not_idled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch, EnumRuntimePresence.DOWN)

    agent._maybe_idle_converge()

    jobs = _jobs(tmp_path)
    assert len(jobs) == 1
    assert jobs[0].command["requested_by"] == IDLE_RESTORE_REQUESTER
    assert jobs[0].command["runtime_lane"] == "dev"
    fake = agent.__dict__["executor"]
    assert isinstance(fake, _FakeExecutor)
    assert "rebuild_scope" in fake.calls

    # One restore per RESTORE_RETRY_AFTER, not one per idle interval.
    agent._idle_converge_last_check = None
    agent._maybe_idle_converge()
    assert len(_jobs(tmp_path)) == 1


def test_a_frozen_down_lane_is_not_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch, EnumRuntimePresence.DOWN)
    agent._lane_flags = ModelLaneFlags(
        freeze=ModelLaneFreeze(until=NOW + timedelta(days=1), reason="test")
    )

    agent._maybe_idle_converge()

    assert _jobs(tmp_path) == []


def test_an_unreadable_presence_never_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch, EnumRuntimePresence.UNKNOWN)
    agent._idle_read_running_ref = lambda repo: "b" * 40  # up to date

    agent._maybe_idle_converge()

    assert _jobs(tmp_path) == []
