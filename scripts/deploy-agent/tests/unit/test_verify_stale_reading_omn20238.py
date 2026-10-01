# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Re-take later failed readings after an earlier target settles (OMN-20238).

Measured on dev-202: runtime's declared-start wait took 321.4s while the
up-front effects failure aged. Docker then reported effects healthy, so the
stale failure skipped the start wait and recreated a healthy container.
The bash health-wait helpers run for real against container-specific fake
Docker states; only curl, compose and the sleeps are faked.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus
from deploy_agent.executor import (
    DeployExecutor,
    VerificationFailedError,
    lane_config_for,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPERS = REPO_ROOT / "scripts" / "runtime_build" / "runtime_health_wait.sh"
RUNTIME_CONTAINER_ID = "a" * 64
EFFECTS_CONTAINER_ID = "b" * 64
HEALTHY_BODY = json.dumps(
    {
        "status": "healthy",
        "details": {"is_running": True, "config_prefetch_status": "skipped"},
    }
)
DECLARED_RUNTIME_HEALTHCHECK = "1800000000000 30000000000 5 10000000000"

_FAKE_DOCKER = """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == "inspect" ]] || { echo "unsupported: $1" >&2; exit 2; }
fmt="$3"
# The helpers inspect both containers. Runtime is starting; effects is healthy.
state_var="FAKE_STATE_${4}"
state="${!state_var}"
if [[ -z "${state}" ]]; then exit 1; fi
case "${fmt}" in
    *'printf "%d %d %d %d" .Config.Healthcheck.StartPeriod'*) printf '%s\\n' "${FAKE_HEALTHCHECK}" ;;
    *State.Status*) printf '%s %s\\n' "${state}" "t0" ;;
    *State.StartedAt*) printf '%s\\n' "t0" ;;
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
    """Runtime fails N probes; effects follows an independent outcome script."""

    def __init__(
        self,
        *,
        fake_bin: Path,
        runtime_fails: int,
        effects_health: list[str],
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> None:
        self.fake_bin = fake_bin
        self.runtime_fails = runtime_fails
        self.effects_health = effects_health
        self.cmds: list[list[str]] = []
        self.runtime_probes = 0
        self.effects_probes = 0
        self.runtime_port, self.effects_port = (
            port for _service, port in lane_config_for(lane).runtime_health_targets
        )
        self.docker_env = {
            "FAKE_HEALTHCHECK": DECLARED_RUNTIME_HEALTHCHECK,
            f"FAKE_STATE_{RUNTIME_CONTAINER_ID}": "running starting",
            f"FAKE_STATE_{EFFECTS_CONTAINER_ID}": "running healthy",
        }

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
                return _completed(cmd, stdout=f"{RUNTIME_CONTAINER_ID}\n")
            if "label=com.docker.compose.service=runtime-effects" in cmd:
                return _completed(cmd, stdout=f"{EFFECTS_CONTAINER_ID}\n")
            return _completed(cmd)
        if "omnidash_analytics" in cmd:
            return _completed(cmd, stdout="t\n")
        if _is_compose_recreate(cmd):
            return _completed(cmd)
        if f"http://localhost:{self.runtime_port}/health" in cmd:
            self.runtime_probes += 1
            if self.runtime_probes <= self.runtime_fails:
                return _completed(
                    cmd, returncode=7, stderr="curl: (7) Failed to connect"
                )
            return _completed(cmd, stdout=HEALTHY_BODY)
        if f"http://localhost:{self.effects_port}/health" in cmd:
            index = min(self.effects_probes, len(self.effects_health) - 1)
            self.effects_probes += 1
            if self.effects_health[index] == "fail":
                return _completed(
                    cmd,
                    returncode=56,
                    stderr="curl: (56) Recv failure: Connection reset by peer",
                )
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
    from deploy_agent import executor as executor_mod

    assert HELPERS.is_file(), HELPERS
    monkeypatch.setattr(executor_mod, "RUNTIME_HEALTH_WAIT_HELPERS", HELPERS)
    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", lambda _seconds: None)


@pytest.mark.unit
def test_stale_reading_a_healthy_effects_is_not_recreated_after_the_runtime_wait(
    fake_bin: Path,
) -> None:
    fake = _StartingLane(
        fake_bin=fake_bin, runtime_fails=3, effects_health=["fail", "healthy"]
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=fake):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(fake.cmds) == []
    assert executor.verify_recreate == []
    assert all(c.status == "pass" for c in checks)
    effects = next(c for c in checks if c.service == "runtime-effects")
    assert "OMN-20238" in effects.detail
    assert fake.runtime_probes == 4
    assert fake.effects_probes == 2


@pytest.mark.unit
def test_stale_reading_still_failing_effects_is_recreated_once(fake_bin: Path) -> None:
    fake = _StartingLane(
        fake_bin=fake_bin, runtime_fails=3, effects_health=["fail", "fail", "healthy"]
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=fake):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(fake.cmds) == ["runtime-effects"]
    assert [r.service for r in executor.verify_recreate] == ["runtime-effects"]
    assert all(c.status == "pass" for c in checks)
    recreate_index = next(
        i for i, cmd in enumerate(fake.cmds) if _is_compose_recreate(cmd)
    )
    # A still-failing FRESH reading must precede the remedy, not follow it.
    assert (
        sum(
            f"http://localhost:{fake.effects_port}/health" in cmd
            for cmd in fake.cmds[:recreate_index]
        )
        == 2
    )
    assert fake.runtime_probes == 4
    assert fake.effects_probes == 3


@pytest.mark.unit
def test_stale_reading_no_retake_when_no_earlier_target_failed(fake_bin: Path) -> None:
    fake = _StartingLane(
        fake_bin=fake_bin, runtime_fails=0, effects_health=["fail", "healthy"]
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=fake):
        checks = executor.verify(on_phase_update=_noop_phase_update)

    assert _recreated_services(fake.cmds) == ["runtime-effects"]
    assert [r.service for r in executor.verify_recreate] == ["runtime-effects"]
    assert all(c.status == "pass" for c in checks)
    assert fake.runtime_probes == 1
    assert fake.effects_probes == 2


@pytest.mark.unit
def test_stale_reading_governed_lane_retakes_but_never_recreates(
    fake_bin: Path,
) -> None:
    lane = EnumRuntimeLane.STABILITY_TEST
    fake = _StartingLane(
        fake_bin=fake_bin,
        runtime_fails=1,
        effects_health=["fail", "healthy"],
        lane=lane,
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_noop_phase_update, lane=lane)

    assert _recreated_services(fake.cmds) == []
    assert executor.verify_recreate == []
    assert not [cmd for cmd in fake.cmds if cmd[0] == "bash"]
    runtime_service, effects_service = (
        service for service, _port in lane_config_for(lane).runtime_health_targets
    )
    statuses = {c.service: c.status for c in executor.health_checks}
    assert statuses[runtime_service] == "fail"
    assert statuses[effects_service] == "pass"
    assert fake.runtime_probes == 1
    assert fake.effects_probes == 2
