# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A stability runtime still inside its declared start budget is waited for.

OMN-24562. ``DeployExecutor.verify`` waited out a runtime Docker still reports
``starting`` (OMN-19374) only on the dev lane: the wait sat behind
``VERIFY_RECREATE_LANES``. A stability runtime declares the same 1800 s start
period and takes about five minutes to answer ``:18085/health``, so its one probe
failed and the agent's automatic rollback restored the retained image.

The wait now runs on the lanes in ``VERIFY_START_WAIT_LANES``; the recreate stays
dev only. The budget is read from the container's own declared healthcheck by the
same ``runtime_health_wait.sh`` helpers, run here as REAL bash against a fake
``docker``. Only the curl probe and the container lookup are faked.

The exception ``verify`` raises past the budget is the one ``agent.py`` turns into
``restore_rollback_point`` (``DeployRolledBackError``); the rollback itself is
covered by ``test_auto_rollback.py``.
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
    VERIFY_RECREATE_LANES,
    VERIFY_START_WAIT_LANES,
    DeployExecutor,
    VerificationFailedError,
    lane_config_for,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPERS = REPO_ROOT / "scripts" / "runtime_build" / "runtime_health_wait.sh"

RUNTIME_CONTAINER_ID = "c" * 64
HEALTHY_BODY = json.dumps(
    {
        "status": "healthy",
        "details": {"is_running": True, "config_prefetch_status": "ok"},
    }
)

# the stability overlay's runtime healthcheck: start_period 1800s, interval 30s,
# retries 5, timeout 10s => a 2000 s budget. Nanoseconds, as docker reports them.
DECLARED_HEALTHCHECK = "1800000000000 30000000000 5 10000000000"
# 30 s start + 1 x (10 s + 10 s) => a 50 s budget: spent well inside the probes.
SHORT_HEALTHCHECK = "30000000000 10000000000 1 10000000000"

_FAKE_DOCKER = """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == "inspect" ]] || { echo "unsupported: $1" >&2; exit 2; }
fmt="$3"
if [[ -z "${FAKE_STATE:-}" ]]; then exit 1; fi
case "${fmt}" in
    *'printf "%d %d %d %d" .Config.Healthcheck.StartPeriod'*) printf '%s\\n' "${FAKE_HEALTHCHECK}" ;;
    '{{.State.Status}}') printf '%s\\n' "${FAKE_STATE%% *}" ;;
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


class _StartingLane:
    """One lane whose main runtime answers its health port only on probe N.

    ``runtime_fails`` is how many probes fail before one passes (``None``: never).
    The container lookup answers the way that lane's real one does: the dev lane
    by compose service label, any other lane by the container name its overlay
    sets. Every ``bash`` command, the only way the executor reaches the health
    wait helpers, runs for real with a fake ``docker`` first on PATH.
    """

    def __init__(
        self,
        *,
        lane: EnumRuntimeLane,
        fake_bin: Path,
        runtime_fails: int | None,
        docker_env: dict[str, str],
        resolve_container: bool = True,
    ) -> None:
        self.lane = lane
        self.runtime_fails = runtime_fails
        self.docker_env = docker_env
        self.resolve_container = resolve_container
        self.fake_bin = fake_bin
        self.cmds: list[list[str]] = []
        self.runtime_probes = 0
        targets = lane_config_for(lane).runtime_health_targets
        (self.runtime_name, self.runtime_port), (_, self.effects_port) = targets

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
            if "--all" not in cmd:
                return _completed(cmd)
            by_label = "label=com.docker.compose.service=omninode-runtime" in cmd
            by_name = f"name=^/{self.runtime_name}$" in cmd
            lookup_matches = by_label if self.lane is EnumRuntimeLane.DEV else by_name
            if lookup_matches and self.resolve_container:
                return _completed(cmd, stdout=f"{RUNTIME_CONTAINER_ID}\n")
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

    def recreates(self) -> list[list[str]]:
        return [cmd for cmd in self.cmds if _is_compose_recreate(cmd)]


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
    """Point the executor at this checkout's helpers and remove only the sleeps."""
    from deploy_agent import executor as executor_mod

    assert HELPERS.is_file(), HELPERS
    monkeypatch.setattr(executor_mod, "RUNTIME_HEALTH_WAIT_HELPERS", HELPERS)
    clock = [0.0]
    monkeypatch.setattr(executor_mod.time, "monotonic", lambda: clock[0])

    def advance(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(executor_mod, "_verify_recreate_sleep", advance)


def _starting(**overrides: str) -> dict[str, str]:
    return {
        "FAKE_HEALTHCHECK": DECLARED_HEALTHCHECK,
        "FAKE_STATE": "running starting",
        **overrides,
    }


WAIT_LANES = [EnumRuntimeLane.DEV, EnumRuntimeLane.STABILITY_TEST]


@pytest.mark.unit
@pytest.mark.parametrize("lane", WAIT_LANES)
def test_a_starting_runtime_is_waited_for_and_passes_within_the_window(
    fake_bin: Path, lane: EnumRuntimeLane
) -> None:
    """The measured shape: the first probe fails, Docker says starting, then it answers."""
    fake = _StartingLane(
        lane=lane, fake_bin=fake_bin, runtime_fails=3, docker_env=_starting()
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=fake):
        checks = executor.verify(on_phase_update=_noop_phase_update, lane=lane)

    runtime = next(
        c for c in checks if c.endpoint.endswith(f":{fake.runtime_port}/health")
    )
    assert runtime.status == "pass"
    # The budget is the container's own declared one, not a number in the agent.
    assert "declared 2000s" in runtime.detail
    assert fake.runtime_probes == 4
    assert fake.recreates() == []
    assert executor.verify_recreate == []


@pytest.mark.unit
@pytest.mark.parametrize("lane", WAIT_LANES)
def test_a_start_that_outlives_its_declared_budget_fails_and_is_not_recreated(
    fake_bin: Path, lane: EnumRuntimeLane
) -> None:
    """Past the window the job fails (agent.py then rolls the lane back); no bounce."""
    fake = _StartingLane(
        lane=lane,
        fake_bin=fake_bin,
        runtime_fails=None,
        docker_env=_starting(FAKE_HEALTHCHECK=SHORT_HEALTHCHECK),
    )
    executor = DeployExecutor()

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError):
            executor.verify(on_phase_update=_noop_phase_update, lane=lane)

    assert fake.recreates() == []
    assert executor.verify_recreate == []
    # Retried inside the 50 s budget, then stopped: more than one probe, bounded.
    assert 1 < fake.runtime_probes < 20


@pytest.mark.unit
def test_the_stability_wait_length_follows_the_declared_budget(
    fake_bin: Path,
) -> None:
    """Nothing is hard-coded: a 50 s declaration allows fewer probes than 2000 s."""
    probes = {}
    for label, healthcheck in (
        ("short", SHORT_HEALTHCHECK),
        ("long", DECLARED_HEALTHCHECK),
    ):
        fake = _StartingLane(
            lane=EnumRuntimeLane.STABILITY_TEST,
            fake_bin=fake_bin,
            runtime_fails=None,
            docker_env=_starting(FAKE_HEALTHCHECK=healthcheck),
        )
        with patch("deploy_agent.executor._run", side_effect=fake):
            with pytest.raises(VerificationFailedError):
                DeployExecutor().verify(
                    on_phase_update=_noop_phase_update,
                    lane=EnumRuntimeLane.STABILITY_TEST,
                )
        probes[label] = fake.runtime_probes
    assert probes["short"] < probes["long"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "docker_env",
    [
        _starting(FAKE_STATE="running unhealthy"),
        _starting(FAKE_STATE="exited none"),
        _starting(FAKE_STATE="running none"),
        _starting(FAKE_STATE=""),
    ],
)
def test_stability_never_recreates_whatever_docker_reports(
    fake_bin: Path, docker_env: dict[str, str]
) -> None:
    """The wait is the only remedy stability gets; a recreate stays dev only."""
    lane = EnumRuntimeLane.STABILITY_TEST
    fake = _StartingLane(
        lane=lane, fake_bin=fake_bin, runtime_fails=1, docker_env=docker_env
    )

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError):
            DeployExecutor().verify(on_phase_update=_noop_phase_update, lane=lane)

    assert fake.recreates() == []


@pytest.mark.unit
def test_an_unresolvable_stability_container_fails_on_the_first_probe(
    fake_bin: Path,
) -> None:
    lane = EnumRuntimeLane.STABILITY_TEST
    fake = _StartingLane(
        lane=lane,
        fake_bin=fake_bin,
        runtime_fails=1,
        docker_env=_starting(),
        resolve_container=False,
    )

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError):
            DeployExecutor().verify(on_phase_update=_noop_phase_update, lane=lane)

    assert fake.runtime_probes == 1
    assert fake.recreates() == []


@pytest.mark.unit
def test_prod_gets_neither_the_wait_nor_the_recreate(fake_bin: Path) -> None:
    lane = EnumRuntimeLane.PROD
    fake = _StartingLane(
        lane=lane, fake_bin=fake_bin, runtime_fails=1, docker_env=_starting()
    )

    with patch("deploy_agent.executor._run", side_effect=fake):
        with pytest.raises(VerificationFailedError):
            DeployExecutor().verify(on_phase_update=_noop_phase_update, lane=lane)

    assert fake.runtime_probes == 1
    assert not [cmd for cmd in fake.cmds if cmd[0] == "bash"]
    assert fake.recreates() == []


@pytest.mark.unit
def test_the_two_fences_are_separate_allowlists() -> None:
    """Widening the wait did not widen the recreate; a new lane joins neither."""
    assert frozenset(WAIT_LANES) == VERIFY_START_WAIT_LANES
    assert frozenset({EnumRuntimeLane.DEV}) == VERIFY_RECREATE_LANES
    assert EnumRuntimeLane.PROD not in VERIFY_START_WAIT_LANES


def _lookup(
    lane: EnumRuntimeLane, name: str, stdout: str
) -> tuple[str, list[list[str]]]:
    seen: list[list[str]] = []

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return _completed(cmd, stdout=stdout)

    with patch("deploy_agent.executor._run", side_effect=run):
        found = DeployExecutor()._compose_service_container(lane, name)
    return found, seen


@pytest.mark.unit
def test_the_dev_lookup_is_the_one_service_label_read_it_always_was() -> None:
    lane = EnumRuntimeLane.DEV
    found, seen = _lookup(lane, "omninode-runtime", "")
    assert found == ""
    assert seen == [
        [
            "docker",
            "ps",
            "--all",
            "-q",
            "--no-trunc",
            "--filter",
            f"label=com.docker.compose.project={lane_config_for(lane).compose_project}",
            "--filter",
            "label=com.docker.compose.service=omninode-runtime",
        ]
    ]


@pytest.mark.unit
def test_a_stability_service_name_still_resolves_by_label_first() -> None:
    """The rollback capture asks for a compose SERVICE on stability; that holds."""
    found, seen = _lookup(
        EnumRuntimeLane.STABILITY_TEST, "omninode-runtime", f"{RUNTIME_CONTAINER_ID}\n"
    )
    assert found == RUNTIME_CONTAINER_ID
    assert len(seen) == 1
    assert "label=com.docker.compose.service=omninode-runtime" in seen[0]


@pytest.mark.unit
def test_a_stability_container_name_resolves_by_name_after_the_label_misses() -> None:
    lane = EnumRuntimeLane.STABILITY_TEST
    name = lane_config_for(lane).runtime_health_targets[0][0]
    found, seen = _lookup(lane, name, "")
    assert found == ""
    assert [cmd[-1] for cmd in seen] == [
        f"label=com.docker.compose.service={name}",
        f"name=^/{name}$",
    ]


@pytest.mark.unit
def test_an_ambiguous_container_match_is_unresolved() -> None:
    found, _ = _lookup(EnumRuntimeLane.STABILITY_TEST, "omninode-runtime", "a\nb\n")
    assert found == ""
