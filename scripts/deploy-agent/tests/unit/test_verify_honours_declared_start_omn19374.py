# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime still inside its declared start budget is waited for, not recreated.

OMN-19374. MEASURED from the dev-lane agent's own journal, 2026-09-23T22:39Z to
2026-09-24T09:01Z: ten consecutive full deploys (jobs 433f0c32, 19e67acc,
594a8aad, 98d35bb2, 0805d076, e4d36317, 2f0fb536, ff4dc5de, d80d3ca9,
5132f2aa) each probed ``:8085/health`` ONCE, 189-276 s after the runtime
containers were started, found it failing, force-recreated ``omninode-runtime``
and watched the NEW container answer after 280-291 s. The one job in the same
window whose gateway leg happened to run long (05743d11, probe at +685 s)
passed without a recreate. The runtime was never broken: it was starting, and
it declares a 1800 s start period. The recreate threw away a container
seconds from ready, restarted its clock, and swapped the container id under
the compose-dev receipt's generation binding.

The fix makes verification consult docker's own reading of that container
before the recreate, through the SAME helpers ``deploy-runtime.sh`` uses
(``scripts/runtime_build/runtime_health_wait.sh``, OMN-18349): while docker
reports it ``starting`` and the wait is inside the declared budget, keep
probing. Anything else -- unhealthy, not running, restarted, absent, no
healthcheck, budget spent -- reaches the recreate exactly as before.

The helpers run as REAL bash against a fake ``docker``; only the curl probe
and the compose recreate are faked.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from deploy_agent.events import (
    EnumRuntimeLane,
    EnumVerifyRecreateOutcome,
    ModelVerifyRecreate,
    Phase,
    PhaseStatus,
)
from deploy_agent.executor import (
    DeployExecutor,
    VerificationFailedError,
    lane_config_for,
)
from deploy_agent.job_state import JobStore

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPERS = REPO_ROOT / "scripts" / "runtime_build" / "runtime_health_wait.sh"

RUNTIME_CONTAINER_ID = "a" * 64
HEALTHY_BODY = json.dumps(
    {
        "status": "healthy",
        "details": {"is_running": True, "config_prefetch_status": "ok"},
    }
)

# The dev-lane runtime's declared healthcheck, read live on .201 2026-09-24:
# StartPeriod 1800s, Interval 30s, Retries 5, Timeout 10s => a 2000 s budget.
DECLARED_RUNTIME_HEALTHCHECK = "1800000000000 30000000000 5 10000000000"

_FAKE_DOCKER = """#!/usr/bin/env bash
set -euo pipefail
# Fake `docker inspect --format FMT ID`, the two reads runtime_health_wait.sh
# makes. FAKE_STATE is "<status> <health>"; empty means no such container.
[[ "$1" == "inspect" ]] || { echo "unsupported: $1" >&2; exit 2; }
fmt="$3"
if [[ -z "${FAKE_STATE:-}" ]]; then exit 1; fi
case "${fmt}" in
    *'printf "%d %d %d %d" .Config.Healthcheck.StartPeriod'*) printf '%s\\n' "${FAKE_HEALTHCHECK}" ;;
    *State.Status*) printf '%s %s\\n' "${FAKE_STATE}" "${FAKE_STARTED_AT:-t0}" ;;
    *State.StartedAt*) printf '%s\\n' "${FAKE_STARTED_AT:-t0}" ;;
    *) echo "unsupported format: ${fmt}" >&2; exit 2 ;;
esac
"""


def _noop_phase_update(phase: Phase, status: PhaseStatus) -> None:
    return None


def _completed(
    cmd: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=cmd, returncode=returncode, stdout=stdout, stderr=stderr
    )


def _is_compose_recreate(cmd: list[str]) -> bool:
    return cmd[:2] == ["docker", "compose"] and "--force-recreate" in cmd


def _recreated_services(cmds: list[list[str]]) -> list[str]:
    return [cmd[-1] for cmd in cmds if _is_compose_recreate(cmd)]


class _StartingLane:
    """The dev lane with a runtime that answers ``:8085`` only on probe N.

    ``runtime_fails`` is how many runtime probes fail before one passes;
    ``None`` means it never passes. Every ``bash`` command -- the only way the
    executor reaches the health-wait helpers -- runs for real, with a fake
    ``docker`` first on PATH reading ``docker_env``.
    """

    def __init__(
        self,
        *,
        fake_bin: Path,
        runtime_fails: int | None,
        docker_env: dict[str, str],
        resolve_container: bool = True,
    ) -> None:
        self.runtime_fails = runtime_fails
        self.docker_env = docker_env
        self.resolve_container = resolve_container
        self.fake_bin = fake_bin
        self.cmds: list[list[str]] = []
        self.runtime_probes = 0
        (self.runtime_port, self.effects_port) = (
            port
            for _service, port in lane_config_for(
                EnumRuntimeLane.DEV
            ).runtime_health_targets
        )

    def __call__(
        self, cmd: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.cmds.append(list(cmd))
        if cmd[0] == "bash":
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env={
                    **os.environ,
                    "PATH": f"{self.fake_bin}{os.pathsep}{os.environ['PATH']}",
                    **self.docker_env,
                },
            )
        if cmd[:2] == ["docker", "ps"]:
            if "label=com.docker.compose.service=omninode-runtime" in cmd:
                return _completed(
                    cmd,
                    stdout=f"{RUNTIME_CONTAINER_ID}\n"
                    if self.resolve_container
                    else "",
                )
            return _completed(cmd)
        if "omnidash_analytics" in cmd:
            return _completed(cmd, stdout="t\n")
        if _is_compose_recreate(cmd):
            return _completed(cmd)
        if f"http://localhost:{self.runtime_port}/health" in cmd:
            self.runtime_probes += 1
            failing = (
                self.runtime_fails is None or self.runtime_probes <= self.runtime_fails
            )
            if failing:
                return _completed(
                    cmd, returncode=7, stderr="curl: (7) Failed to connect"
                )
            return _completed(cmd, stdout=HEALTHY_BODY)
        if f"http://localhost:{self.effects_port}/health" in cmd:
            return _completed(cmd, stdout=HEALTHY_BODY)
        return _completed(cmd, returncode=1, stderr=f"unexpected command: {cmd}")


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return bin_dir


@pytest.fixture(autouse=True)
def _real_helpers_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the executor at this checkout's helpers and remove the waits.

    The sleep between probes is the only thing removed: the helper calls, the
    budget they derive and the probe count are all real.
    """
    from deploy_agent import executor as executor_mod

    assert HELPERS.is_file(), HELPERS
    monkeypatch.setattr(executor_mod, "RUNTIME_HEALTH_WAIT_HELPERS", HELPERS)
    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", lambda _seconds: None)


def _starting(**overrides: str) -> dict[str, str]:
    return {
        "FAKE_HEALTHCHECK": DECLARED_RUNTIME_HEALTHCHECK,
        "FAKE_STATE": "running starting",
        **overrides,
    }


@pytest.mark.unit
def test_a_starting_runtime_is_waited_for_and_never_recreated(fake_bin: Path) -> None:
    """The measured shape: probe fails at +190..276 s, docker says starting."""
    lane = _StartingLane(fake_bin=fake_bin, runtime_fails=3, docker_env=_starting())
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == []
    assert executor.verify_recreate == []
    runtime = next(c for c in checks if c.service == "omninode-runtime")
    assert runtime.status == "pass"
    # The wait is legible on the check itself, so a reader of the terminal
    # event can tell a waited-for start from a first-probe pass.
    assert "declared" in runtime.detail
    assert "2000s" in runtime.detail
    assert lane.runtime_probes == 4


@pytest.mark.unit
def test_the_helpers_are_the_ones_deploy_runtime_sh_uses(fake_bin: Path) -> None:
    """Reuse, not a second copy: every docker reading goes through the helpers."""
    lane = _StartingLane(fake_bin=fake_bin, runtime_fails=1, docker_env=_starting())

    with patch("deploy_agent.executor._run", side_effect=lane):
        DeployExecutor().verify(on_phase_update=_noop_phase_update)

    helper_calls = [cmd for cmd in lane.cmds if cmd[0] == "bash"]
    assert helper_calls, "the wait never consulted the helpers"
    assert all(str(HELPERS) in cmd for cmd in helper_calls)
    called = {cmd[cmd.index(str(HELPERS)) + 1] for cmd in helper_calls}
    assert {
        "runtime_health_budget_seconds",
        "runtime_started_at",
        "runtime_health_state",
        "runtime_health_keep_waiting",
    } <= called
    # No `docker inspect` of the runtime issued from Python: the helper owns it.
    assert not [c for c in lane.cmds if c[:2] == ["docker", "inspect"]]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("docker_env", "why"),
    [
        (_starting(FAKE_STATE="running unhealthy"), "docker says unhealthy"),
        (_starting(FAKE_STATE="exited none"), "the container is not running"),
        (_starting(FAKE_STATE="running none"), "no healthcheck declared"),
        (_starting(FAKE_HEALTHCHECK="0 0 0 0"), "a zero declared budget"),
        (_starting(FAKE_STATE=""), "docker cannot inspect it"),
    ],
)
def test_positive_control_a_dead_runtime_is_still_recreated(
    fake_bin: Path, docker_env: dict[str, str], why: str
) -> None:
    """The recreate the OMN-18640 wedge needs is untouched by the wait."""
    lane = _StartingLane(fake_bin=fake_bin, runtime_fails=1, docker_env=docker_env)
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == ["omninode-runtime"], why
    assert len(executor.verify_recreate) == 1, why


@pytest.mark.unit
def test_positive_control_an_unresolvable_container_is_still_recreated(
    fake_bin: Path,
) -> None:
    lane = _StartingLane(
        fake_bin=fake_bin,
        runtime_fails=1,
        docker_env=_starting(),
        resolve_container=False,
    )

    with patch("deploy_agent.executor._run", side_effect=lane):
        DeployExecutor().verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == ["omninode-runtime"]


@pytest.mark.unit
def test_a_start_that_outlives_its_declared_budget_is_recreated(
    fake_bin: Path,
) -> None:
    """Bounded: a runtime starting forever is recreated once the budget is spent.

    A 50 s budget (30 s start period + 1 x (10 s + 10 s)) at the 10 s poll is
    five probes; the sixth is the recreate's own re-probe.
    """
    lane = _StartingLane(
        fake_bin=fake_bin,
        runtime_fails=None,
        docker_env=_starting(FAKE_HEALTHCHECK="30000000000 10000000000 1 10000000000"),
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=lane):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(lane.cmds) == ["omninode-runtime"]
    probes_before_recreate = 0
    for cmd in lane.cmds:
        if _is_compose_recreate(cmd):
            break
        if f"http://localhost:{lane.runtime_port}/health" in cmd:
            probes_before_recreate += 1
    # The first failing probe plus five inside the budget.
    assert probes_before_recreate == 6


@pytest.mark.unit
@pytest.mark.parametrize("lane", [EnumRuntimeLane.STABILITY_TEST, EnumRuntimeLane.PROD])
def test_a_governed_lane_neither_waits_nor_recreates(
    fake_bin: Path, lane: EnumRuntimeLane
) -> None:
    """The wait sits in front of the dev-only recreate and nowhere else."""
    fake = _StartingLane(fake_bin=fake_bin, runtime_fails=1, docker_env=_starting())
    fake.runtime_port, fake.effects_port = (
        port for _service, port in lane_config_for(lane).runtime_health_targets
    )

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError):
            DeployExecutor().verify(on_phase_update=_noop_phase_update, lane=lane)

    assert _recreated_services(fake.cmds) == []
    assert not [cmd for cmd in fake.cmds if cmd[0] == "bash"]


# --- the job record carries the recreate (AC2's evidence source) -------------


def _recovered_runtime() -> ModelVerifyRecreate:
    return ModelVerifyRecreate(
        service="omninode-runtime",
        lane=EnumRuntimeLane.DEV,
        compose_project="omnibase-infra",
        endpoint="http://localhost:8085/health",
        outcome=EnumVerifyRecreateOutcome.RECOVERED,
        recreate_returncode=0,
        readiness_wait_seconds=280.0,
        readiness_budget_seconds=600,
    )


@pytest.mark.unit
def test_the_terminal_write_persists_the_in_job_recreate(tmp_path: Path) -> None:
    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "full"})

    store.complete(cid, status="success", verify_recreate=[_recovered_runtime()])

    reloaded = store.load(cid)
    assert reloaded is not None
    assert reloaded.status == "success"
    assert [r.outcome for r in reloaded.verify_recreate] == [
        EnumVerifyRecreateOutcome.RECOVERED
    ]


@pytest.mark.unit
def test_the_job_endpoint_serves_the_in_job_recreate(tmp_path: Path) -> None:
    """The lab guard reads ``/job/{cid}``; a field it cannot see does not exist."""
    from deploy_agent.health import _job_handler

    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "full"})
    store.complete(cid, status="success", verify_recreate=[_recovered_runtime()])
    request = SimpleNamespace(
        app={"job_store": store}, match_info={"correlation_id": str(cid)}
    )

    body = json.loads(asyncio.run(_job_handler(request)).text)  # type: ignore[arg-type]

    assert body["status"] == "success"
    assert body["verify_recreate"] == [
        {
            "service": "omninode-runtime",
            "lane": "dev",
            "compose_project": "omnibase-infra",
            "endpoint": "http://localhost:8085/health",
            "outcome": "recovered",
            "recreate_returncode": 0,
            "readiness_wait_seconds": 280.0,
            "readiness_budget_seconds": 600,
            "detail": "",
        }
    ]


@pytest.mark.unit
def test_a_job_with_no_recreate_serves_an_empty_list(tmp_path: Path) -> None:
    from deploy_agent.health import _job_handler

    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "full"})
    store.complete(cid, status="success")
    request = SimpleNamespace(
        app={"job_store": store}, match_info={"correlation_id": str(cid)}
    )

    body = json.loads(asyncio.run(_job_handler(request)).text)  # type: ignore[arg-type]

    assert body["verify_recreate"] == []


@pytest.mark.unit
def test_the_agent_hands_the_recreate_to_both_terminal_writes() -> None:
    """Read from the source: driving a whole job needs a broker and a lane."""
    agent_source = (
        Path(__file__).resolve().parents[2] / "deploy_agent" / "agent.py"
    ).read_text(encoding="utf-8")
    assert agent_source.count("verify_recreate=self.executor.verify_recreate,") == 3, (
        "the success write, the failure write and the terminal event"
    )
