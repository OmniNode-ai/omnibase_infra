# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A compose up that did not converge fails its phase (OMN-17427).

Regression scenario: ``docker compose -p
omnibase-infra --profile core up -d --pull always`` exited 1 on a container-name
conflict for infisical, after printing ``Recreate`` for redpanda. Every expected
container (postgres, redpanda, valkey) was still running, so the agent logged
"every expected service is running; phase core passes on live state" and
recorded SUCCESS. The redpanda recreate never happened (its container kept the
old ``--memory 12G`` config while compose declared 6G) and infisical was never
created under compose. The same pass repeated six times in six hours.

Running is not converged. After a killed or non-zero up, the phase passes only
when every service in scope carries the ``com.docker.compose.config-hash`` label
compose renders for it now.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

import pytest
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import DeployExecutor

# Section 1 of the lane brief, verbatim.
TODAY_STDERR = """\
 Container omnibase-infra-valkey Running
 Container omnibase-infra-postgres Running
 Container omnibase-infra-redpanda Recreate
 Container omnibase-infra-infisical Creating
 Container omnibase-infra-keycloak Running
 Container omnibase-infra-infisical Error response from daemon: Conflict. The container name "/omnibase-infra-infisical" is already in use by container "436f174b56e2f9224d5b85fb47d155273e6bf3efbd0c36cb541e2da8fafecc2a". You have to remove (or rename) that container to be able to reuse that name.
Error response from daemon: Conflict. The container name "/omnibase-infra-infisical" is already in use by container "436f174b56e2f9224d5b85fb47d155273e6bf3efbd0c36cb541e2da8fafecc2a". You have to remove (or rename) that container to be able to reuse that name."""

HASH_A = "a" * 64
HASH_B = "b" * 64

CORE_DECLARED = {
    "postgres": "1" * 64,
    "redpanda": HASH_B,
    "valkey": "2" * 64,
    "infisical": "3" * 64,
}


class _Docker:
    """Answers the docker commands `_compose_up` makes, and records them."""

    def __init__(
        self,
        *,
        declared: dict[str, str],
        labels: dict[str, str],
        up_returncode: int = 0,
        up_stderr: str = "",
        up_timeout: bool = False,
        hash_returncode: int = 0,
    ) -> None:
        self.declared = declared
        self.labels = labels
        self.up_returncode = up_returncode
        self.up_stderr = up_stderr
        self.up_timeout = up_timeout
        self.hash_returncode = hash_returncode
        self.calls: list[list[str]] = []

    def hash_calls(self) -> list[list[str]]:
        return [c for c in self.calls if "config" in c and "--hash" in c]

    def run(self, cmd: list[str], timeout: int, **kwargs: Any) -> Any:
        self.calls.append(list(cmd))
        if "config" in cmd and "--hash" in cmd:
            if self.hash_returncode != 0:
                return subprocess.CompletedProcess(
                    cmd, self.hash_returncode, "", 'service "nope" has no image'
                )
            wanted = cmd[cmd.index("--hash") + 1]
            names = list(self.declared) if wanted == "*" else wanted.split(",")
            out = "".join(
                f"{n} {self.declared[n]}\n" for n in names if n in self.declared
            )
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if cmd[:2] == ["docker", "ps"]:
            prefix = "label=com.docker.compose.service="
            service = next(c.removeprefix(prefix) for c in cmd if c.startswith(prefix))
            label = self.labels.get(service, "")
            return subprocess.CompletedProcess(
                cmd, 0, f"{label}\n" if label else "", ""
            )
        if "up" in cmd:
            if self.up_timeout:
                raise subprocess.TimeoutExpired(cmd, timeout)
            return subprocess.CompletedProcess(
                cmd, self.up_returncode, "", self.up_stderr
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    from deploy_agent import executor as executor_mod

    # Every expected container reports running: the live state of section 1.
    monkeypatch.setattr(
        executor_mod, "verify_containers_up", lambda *a, **k: (True, [])
    )
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})
    monkeypatch.setattr(executor_mod, "finish_recreate_renames", lambda lane: [])
    monkeypatch.setattr(executor_mod, "remove_recreate_leftovers", lambda *a, **k: None)
    monkeypatch.setattr(
        executor_mod.DeployExecutor,
        "observe_deps_convergence",
        lambda self, lane: None,
    )
    monkeypatch.setattr(
        executor_mod.DeployExecutor,
        "_ensure_runtime_migrations_ready",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        executor_mod.DeployExecutor,
        "rerun_changed_oneshots",
        lambda self, **kwargs: [],
    )
    return monkeypatch


def _install(monkeypatch: pytest.MonkeyPatch, docker: _Docker) -> None:
    from deploy_agent import executor as executor_mod

    monkeypatch.setattr(executor_mod, "_run", docker.run)

    def _deps_recreate(self: DeployExecutor, cmd: list[str], **kwargs: Any) -> str:
        # The supervisor's COMPLETED ending, shaped as it reports it.
        result = docker.run(cmd, timeout=0)
        if result.returncode == 0:
            return ""
        return (
            f"docker compose up for phase core exited {result.returncode}: "
            f"{result.stderr.strip()[-2000:]}"
        )

    monkeypatch.setattr(
        executor_mod.DeployExecutor, "_supervised_deps_recreate", _deps_recreate
    )


def _core_up(phases: list[tuple[Phase, PhaseStatus]]) -> None:
    DeployExecutor()._compose_up(
        Phase.CORE,
        Scope.CORE,
        [],
        lambda phase, status: phases.append((phase, status)),
        lane=EnumRuntimeLane.DEV,
    )


@pytest.mark.unit
def test_todays_unconverged_core_up_fails_the_phase(
    patched: pytest.MonkeyPatch,
) -> None:
    """RED at origin/dev 39e52ccb2: the phase passed on 'every expected service is running'."""
    docker = _Docker(
        declared=CORE_DECLARED,
        labels={
            "postgres": CORE_DECLARED["postgres"],
            "redpanda": HASH_A,
            "valkey": CORE_DECLARED["valkey"],
        },
        up_returncode=1,
        up_stderr=TODAY_STDERR,
    )
    _install(patched, docker)
    phases: list[tuple[Phase, PhaseStatus]] = []

    with pytest.raises(RuntimeError) as excinfo:
        _core_up(phases)

    message = str(excinfo.value)
    assert message.startswith("Docker compose up failed:")
    assert "already in use by container" in message
    assert f"redpanda running={HASH_A[:12]} declared={HASH_B[:12]}" in message
    assert "infisical running=none declared=" in message
    assert "postgres running=" not in message
    assert "valkey running=" not in message
    assert (Phase.CORE, PhaseStatus.SUCCESS) not in phases
    # The render is the profile's, with the same project and profile as the up.
    (hash_call,) = docker.hash_calls()
    assert hash_call[-1] == "*"
    assert hash_call[
        hash_call.index("--profile") : hash_call.index("--profile") + 2
    ] == ["--profile", "core"]


@pytest.mark.unit
def test_nonzero_up_with_every_service_converged_passes(
    patched: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    docker = _Docker(
        declared=CORE_DECLARED,
        labels=dict(CORE_DECLARED),
        up_returncode=1,
        up_stderr=TODAY_STDERR,
    )
    _install(patched, docker)
    phases: list[tuple[Phase, PhaseStatus]] = []

    with caplog.at_level(logging.WARNING, logger="deploy_agent.executor"):
        _core_up(phases)

    assert (Phase.CORE, PhaseStatus.SUCCESS) in phases
    assert any(
        "every service in scope is at its declared configuration" in r.getMessage()
        for r in caplog.records
    )


def _runtime_up(phases: list[tuple[Phase, PhaseStatus]]) -> None:
    DeployExecutor()._compose_up(
        Phase.RUNTIME,
        Scope.RUNTIME,
        ["omninode-runtime"],
        lambda phase, status: phases.append((phase, status)),
        lane=EnumRuntimeLane.DEV,
    )


@pytest.mark.unit
def test_ceiling_kill_with_scope_converged_passes(
    patched: pytest.MonkeyPatch,
) -> None:
    """OMN-18057 kept: a killed up whose services all converged passes with its warning."""
    declared = {"omninode-runtime": HASH_B}
    docker = _Docker(declared=declared, labels=dict(declared), up_timeout=True)
    _install(patched, docker)
    phases: list[tuple[Phase, PhaseStatus]] = []

    _runtime_up(phases)

    assert (Phase.RUNTIME, PhaseStatus.SUCCESS) in phases
    (hash_call,) = docker.hash_calls()
    assert hash_call[-1] == "omninode-runtime"


@pytest.mark.unit
def test_ceiling_kill_with_one_unconverged_service_fails(
    patched: pytest.MonkeyPatch,
) -> None:
    docker = _Docker(
        declared={"omninode-runtime": HASH_B},
        labels={"omninode-runtime": HASH_A},
        up_timeout=True,
    )
    _install(patched, docker)
    phases: list[tuple[Phase, PhaseStatus]] = []

    with pytest.raises(RuntimeError) as excinfo:
        _runtime_up(phases)

    message = str(excinfo.value)
    assert message.startswith("Docker compose up failed:")
    assert "ceiling" in message
    assert f"omninode-runtime running={HASH_A[:12]} declared={HASH_B[:12]}" in message
    assert (Phase.RUNTIME, PhaseStatus.SUCCESS) not in phases


@pytest.mark.unit
def test_config_hash_failure_after_compose_error_fails_closed(
    patched: pytest.MonkeyPatch,
) -> None:
    docker = _Docker(
        declared=CORE_DECLARED,
        labels=dict(CORE_DECLARED),
        up_returncode=1,
        up_stderr=TODAY_STDERR,
        hash_returncode=15,
    )
    _install(patched, docker)
    phases: list[tuple[Phase, PhaseStatus]] = []

    with pytest.raises(RuntimeError) as excinfo:
        _core_up(phases)

    message = str(excinfo.value)
    assert message.startswith("Docker compose up failed:")
    assert "`compose config --hash` exited 15" in message
    assert "has no image" in message
    assert (Phase.CORE, PhaseStatus.SUCCESS) not in phases


@pytest.mark.unit
@pytest.mark.parametrize("phase_kind", ["core", "runtime"])
def test_clean_up_makes_no_config_hash_call(
    patched: pytest.MonkeyPatch, phase_kind: str
) -> None:
    # Labels deliberately disagree: a clean exit must not even look.
    docker = _Docker(
        declared={**CORE_DECLARED, "omninode-runtime": HASH_B},
        labels={"redpanda": HASH_A, "omninode-runtime": HASH_A},
    )
    _install(patched, docker)
    phases: list[tuple[Phase, PhaseStatus]] = []

    if phase_kind == "core":
        _core_up(phases)
    else:
        _runtime_up(phases)

    assert docker.hash_calls() == []
    assert not [c for c in docker.calls if c[:2] == ["docker", "ps"]]
    assert phases[-1][1] is PhaseStatus.SUCCESS
