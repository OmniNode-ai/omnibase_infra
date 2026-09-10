# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Phase.RUNTIME compose-up ceiling must not bypass container recovery (OMN-18057).

MEASURED, 2026-09-08 (ROLLING_WORK_LEDGER.md:5076). Command 23edaf62 ran the
ten-service runtime force-recreate against ``PHASE_TIMEOUTS[Phase.RUNTIME] =
300``. The runtime container's own log records ``Bootstrap time: 249.600s`` and
the lane first bound :8085 at t+321s, so the kill landed **21 seconds before the
gating dependency became serviceable**. ``subprocess.run`` raised
``TimeoutExpired`` straight out of ``_compose_up``, which is upstream of the
``verify_containers_up`` + per-container ``up -d --no-deps`` recovery that the
same method already runs when compose exits non-zero. Three services were left
in ``Created``, :8086 stayed down, and the OCC mint was down fleet-wide.

The control below is the timeout half of that: a stub compose that sleeps past a
small injected ceiling while the lane sits in ``Created``. It asserts the
recovery ran and that the raised error names the residue -- not that the process
died at the compose boundary.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from deploy_agent.events import (
    DEV_LANE_ONLY_RUNTIME_SERVICES,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor

# The three services the live incident left in Created (ledger row :5076).
STUCK_SERVICES = (
    "runtime-effects",
    "runtime-worker",
    "omninode-contract-resolver",
)
RUNNING_SERVICES = (
    "omninode-runtime",
    "agent-actions-consumer",
    "skill-lifecycle-consumer",
    "context-audit-consumer",
    "intelligence-migration",
    "intelligence-api",
    "autoheal",
    # OMN-18108: a DEV runtime deploy also targets the lane's own services, so
    # the modelled lane has to contain them. They came up; the three above did
    # not. A service absent from this map reads as unresolvable, which would
    # make this fixture assert recovery of services the incident never touched.
    *DEV_LANE_ONLY_RUNTIME_SERVICES,
)

# The stub compose sleeps this long; the injected ceiling is far below it, so
# the wait is a REAL subprocess timeout rather than a hand-raised exception.
_STUB_COMPOSE_SLEEP_SECONDS = 30
_INJECTED_CEILING_SECONDS = 1

# Captured before any monkeypatching so the stub's own child process is never
# routed through the recovery spy installed below.
_REAL_SUBPROCESS_RUN = subprocess.run


def _is_compose_up(cmd: list[str]) -> bool:
    return "up" in cmd and "--force-recreate" in cmd


class _ComposeStub:
    """Records calls and sleeps past the injected ceiling on the up command."""

    def __init__(self) -> None:
        self.up_calls: list[list[str]] = []
        self.timeouts_seen: list[int] = []
        self.recovery_calls: list[str] = []
        self.verify_calls: list[list[str]] = []

    def run(self, cmd: list[str], timeout: int, **kwargs: Any) -> Any:
        if _is_compose_up(cmd):
            self.up_calls.append(list(cmd))
            self.timeouts_seen.append(timeout)
            # A real child process that outlives a real (small) ceiling.
            return _REAL_SUBPROCESS_RUN(
                [
                    sys.executable,
                    "-c",
                    f"import time; time.sleep({_STUB_COMPOSE_SLEEP_SECONDS})",
                ],
                timeout=_INJECTED_CEILING_SECONDS,
                capture_output=True,
                text=True,
                check=False,
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")


@pytest.fixture
def compose_stub(monkeypatch: pytest.MonkeyPatch) -> _ComposeStub:
    from deploy_agent import executor as executor_mod

    stub = _ComposeStub()
    monkeypatch.setattr(executor_mod, "_run", stub.run)

    # The migration preflight is a separate, already-covered gate; it is not
    # what this control exercises.
    monkeypatch.setattr(
        executor_mod.DeployExecutor,
        "_ensure_runtime_migrations_ready",
        lambda self, **kwargs: None,
    )

    def _states(
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> dict[str, tuple[str, int | None]]:
        states: dict[str, tuple[str, int | None]] = dict.fromkeys(
            RUNNING_SERVICES, ("running", None)
        )
        states.update(dict.fromkeys(STUCK_SERVICES, ("created", None)))
        return states

    monkeypatch.setattr(executor_mod, "_compose_service_states", _states)

    def _recover(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        stub.recovery_calls.append(cmd[-1])
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    # Only the per-container recovery call inside _compose_up goes through the
    # module's bare ``subprocess.run``; everything else uses ``_run``.
    monkeypatch.setattr(executor_mod.subprocess, "run", _recover)

    # Keep the REAL verify_containers_up logic, shrink only its poll window so
    # the control is a unit test rather than a three-minute wait.
    real_verify = executor_mod.verify_containers_up

    def _fast_verify(
        expected: list[str],
        timeout_s: int = 120,
        *,
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> tuple[bool, list[str]]:
        stub.verify_calls.append(list(expected))
        return real_verify(expected, timeout_s=1, lane=lane)

    monkeypatch.setattr(executor_mod, "verify_containers_up", _fast_verify)
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})
    return stub


@pytest.mark.unit
def test_runtime_compose_up_timeout_runs_container_recovery(
    compose_stub: _ComposeStub,
) -> None:
    """A blown compose-up ceiling must reach the SAME recovery a non-zero exit does.

    RED before OMN-18057: ``subprocess.TimeoutExpired`` escapes ``_compose_up``,
    ``verify_containers_up`` is never called, and no per-container recovery is
    attempted -- exactly the live 23edaf62 residue.
    """
    executor = DeployExecutor()
    phases: list[tuple[Phase, PhaseStatus]] = []

    with pytest.raises(RuntimeError) as excinfo:
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            [],
            lambda phase, status: phases.append((phase, status)),
            lane=EnumRuntimeLane.DEV,
        )

    # The error is a phase verdict, not a raw process death.
    assert not isinstance(excinfo.value, subprocess.TimeoutExpired)
    message = str(excinfo.value)
    for service in STUCK_SERVICES:
        assert service in message, f"{service} missing from {message!r}"

    # The recovery the success path already runs must have been reached.
    assert compose_stub.verify_calls, (
        "verify_containers_up was never called after the compose-up ceiling blew"
    )
    # ...and attempted for every stuck service.
    assert sorted(compose_stub.recovery_calls) == sorted(STUCK_SERVICES)

    # The phase is reported FAILED, never left IN_PROGRESS.
    assert (Phase.RUNTIME, PhaseStatus.IN_PROGRESS) in phases
    assert (Phase.RUNTIME, PhaseStatus.SUCCESS) not in phases


@pytest.mark.unit
def test_runtime_compose_up_timeout_records_residue(
    compose_stub: _ComposeStub,
) -> None:
    """The residue (service names + states) must be recorded for the terminal event."""
    executor = DeployExecutor()

    with pytest.raises(RuntimeError):
        executor._compose_up(
            Phase.RUNTIME,
            Scope.RUNTIME,
            [],
            lambda phase, status: None,
            lane=EnumRuntimeLane.DEV,
        )

    residue = {item.service: item.state for item in executor.container_residue}
    assert sorted(residue) == sorted(STUCK_SERVICES)
    assert set(residue.values()) == {"created"}
