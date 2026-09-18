# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The executor's deps phase: derived ceiling, deferral, convergence (OMN-18692).

``test_deps_recreate_supervisor_omn18692.py`` drives the supervisor in
isolation. This module drives it THROUGH ``DeployExecutor._compose_up``, which
is where the three things that matter in production actually meet: the ceiling
comes from the lane's own compose files, the deferral refuses before anything
is spawned, and a command the supervisor had to end is followed by the
deps-only convergence rather than by a phase verdict alone.

Every test here carries the ``deps_recreate`` marker, which opts out of the
conftest stub that routes the deps phase through ``_run`` for tests that are
not about it. These ARE about it, so they drive the real method and stub only
the two seams that reach the machine: ``_spawn_compose`` and
``_compose_service_states``.

THE GOLDEN AND ERROR CHAINS (the 2026-09-10 paired-chain ruling) are the last
two tests: one drives a healthy deps recreate all the way to the terminal
payload and asserts the receipt names the wait; the other drives the 2026-09-18
shape -- a recreate the supervisor had to end with the broker absent -- all the
way to the same payload, and asserts the broker came back and the receipt names
the kill.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent.events import (
    SCOPE_SERVICES,
    EnumRecreateOutcome,
    EnumRuntimeLane,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import PHASE_TIMEOUTS, DeployExecutor
from deploy_agent.host_conditions import EnumBuildCacheState, ModelHostConditions
from deploy_agent.publisher import build_completion_payload
from deploy_agent.recreate_supervisor import HostContentionDeferralError

pytestmark = [pytest.mark.unit, pytest.mark.deps_recreate]

DEPS = tuple(SCOPE_SERVICES[Scope.CORE])
RUNNING = dict.fromkeys(DEPS, ("running", None))
#: The measured residue: redpanda and valkey gone, postgres half-recreated.
HALF_RECREATED: dict[str, tuple[str, int | None]] = {"postgres": ("created", None)}


def _host(saturation: float) -> ModelHostConditions:
    return ModelHostConditions(
        load1=saturation * 32,
        cpu_count=32,
        saturation=saturation,
        cache_state=EnumBuildCacheState.WARM,
        contention_multiplier=1.0,
        cache_multiplier=1.0,
    )


class _SpawnedCompose:
    """A stand-in for the compose CLI that exits ``returncode`` immediately."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode: int | None = returncode
        self.terminate_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:  # pragma: no cover - not reached when it exits
        self.terminate_calls += 1

    def kill(self) -> None:  # pragma: no cover
        pass

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


class _NeverExitingCompose(_SpawnedCompose):
    """A compose CLI that hangs mid-recreate until it is signalled."""

    def __init__(self) -> None:
        super().__init__(returncode=0)
        self.returncode = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15


@pytest.fixture
def lane_state() -> dict[str, dict[str, tuple[str, int | None]]]:
    """Mutable lane state the stubs read and the convergence writes."""
    return {"states": dict(RUNNING)}


@pytest.fixture
def executor_seams(
    monkeypatch: pytest.MonkeyPatch,
    lane_state: dict[str, dict[str, tuple[str, int | None]]],
) -> dict[str, Any]:
    """Stub only what reaches the machine; leave every decision live."""
    from deploy_agent import executor as executor_mod
    from deploy_agent import recreate_supervisor

    recorded: dict[str, Any] = {"spawned": [], "runs": [], "host": _host(0.1)}

    def _states(
        lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
    ) -> Mapping[str, tuple[str, int | None]]:
        return dict(lane_state["states"])

    def _spawn(cmd: list[str], **kwargs: Any) -> Any:
        recorded["spawned"].append(list(cmd))
        return recorded.get("process") or _SpawnedCompose()

    def _run(cmd: list[str], timeout: int, **kwargs: Any) -> Any:
        recorded["runs"].append(list(cmd))
        # The deps-only convergence brings the lane back, exactly as the hand
        # recovery did at 13:35:56Z on 2026-09-18.
        if "up" in cmd and "--force-recreate" not in cmd:
            lane_state["states"] = dict(RUNNING)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(executor_mod, "_compose_service_states", _states)
    monkeypatch.setattr(executor_mod, "_spawn_compose", _spawn)
    monkeypatch.setattr(executor_mod, "_run", _run)
    monkeypatch.setattr(executor_mod, "_compose_env", lambda *a, **k: {})
    monkeypatch.setattr(executor_mod, "probe_host_conditions", lambda: recorded["host"])
    monkeypatch.setattr(
        recreate_supervisor, "probe_host_conditions", lambda: recorded["host"]
    )
    return recorded


@pytest.fixture
def fast_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compress the supervisor's WAITS, never its decisions.

    Only wall-clock policy numbers move: the poll cadence, the stall window,
    the deferral wait, and the ceiling itself. Which branch the supervisor
    takes, in what order, against what lane state, is untouched -- and the
    executor reads the first three at its call site precisely so a test can
    move them without replacing the function under test.

    The ceiling is compressed by substituting the DERIVATION, because it cannot
    be compressed by moving a constant: it is read from the real compose file,
    where ``postgres`` declares ``start_period: 180s``, so the smallest honest
    production ceiling is still minutes long. ``model_seconds`` is left at the
    real derived value so a test can still assert what the production ceiling
    is derived FROM. That the derivation itself reads the live compose model
    and the live machine is asserted separately, above, with no compression at
    all -- splitting it this way is what keeps both claims provable in a suite
    that has to finish.
    """
    from deploy_agent import executor as executor_mod
    from deploy_agent import recreate_supervisor

    monkeypatch.setattr(recreate_supervisor, "SUPERVISOR_POLL_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(recreate_supervisor, "MID_RECREATE_STALL_SECONDS", 0.05)
    monkeypatch.setattr(recreate_supervisor, "DEPS_RECREATE_DEFER_POLL_SECONDS", 0.001)
    monkeypatch.setattr(
        recreate_supervisor, "DEPS_RECREATE_DEFER_MAX_WAIT_SECONDS", 0.01
    )

    real = executor_mod.deps_compose_up_budget

    def _compressed(
        lane: EnumRuntimeLane, expected_services: list[str]
    ) -> recreate_supervisor.ModelDepsRecreateBudget:
        derived = real(lane, expected_services)
        return derived.model_copy(
            update={"timeout_seconds": 1, "hard_upper_bound_seconds": 3}
        )

    monkeypatch.setattr(executor_mod, "deps_compose_up_budget", _compressed)


def _deps_up(executor: DeployExecutor, phases: list[tuple[Phase, PhaseStatus]]) -> None:
    executor._compose_up(
        Phase.CORE,
        Scope.CORE,
        [],
        lambda phase, status: phases.append((phase, status)),
        lane=EnumRuntimeLane.DEV,
    )


def test_the_deps_ceiling_is_derived_from_the_compose_model_omn18692(
    executor_seams: dict[str, Any],
) -> None:
    """AC2: the flat phase bound no longer bounds the deps compose-up.

    SUPERSEDES ``test_core_compose_up_keeps_its_flat_phase_bound`` in
    ``test_compose_budget_omn18057.py``, which asserted the opposite. That
    assertion was correct when it was written -- OMN-18057 changed the runtime
    phase and deliberately left the core phase alone -- and the flat 300 it
    pinned is the number that removed the dev-lane broker nine days later.
    """
    from deploy_agent import executor as executor_mod

    budget = executor_mod.deps_compose_up_budget(
        EnumRuntimeLane.DEV, list(SCOPE_SERVICES[Scope.CORE])
    )

    assert budget.timeout_seconds > PHASE_TIMEOUTS[Phase.CORE]
    # Read from the compose model, not from a constant: postgres declares
    # `start_period: 180s` and `omninode-runtime` gates on it via
    # `depends_on: {condition: service_healthy}`.
    assert budget.source_service == "postgres"
    assert budget.model_seconds >= 180
    assert "compose model" in budget.describe()


def test_the_ceiling_moves_with_the_machine_omn18692(
    executor_seams: dict[str, Any],
) -> None:
    """The incident's saturation must widen it; an idle host must not."""
    from deploy_agent import executor as executor_mod

    executor_seams["host"] = _host(0.1)
    quiet = executor_mod.deps_compose_up_budget(
        EnumRuntimeLane.DEV, list(SCOPE_SERVICES[Scope.CORE])
    )
    executor_seams["host"] = _host(28.42 / 32)
    loaded = executor_mod.deps_compose_up_budget(
        EnumRuntimeLane.DEV, list(SCOPE_SERVICES[Scope.CORE])
    )

    assert loaded.timeout_seconds > quiet.timeout_seconds


def test_a_loaded_host_refuses_before_anything_is_spawned_omn18692(
    executor_seams: dict[str, Any],
    fast_supervisor: None,
) -> None:
    """AC1: the refusal happens BEFORE the removal, so the lane is untouched.

    The falsifier is the spawn list, not the exception: an implementation that
    raised after starting compose would satisfy a ``pytest.raises`` and would
    still have destroyed the lane.
    """
    executor_seams["host"] = _host(3.0)
    executor = DeployExecutor()

    with pytest.raises(HostContentionDeferralError) as excinfo:
        _deps_up(executor, [])

    assert executor_seams["spawned"] == [], (
        "the deps recreate was started despite the deferral; the whole value "
        "of refusing is that nothing has been removed yet"
    )
    assert "THE LANE WAS NOT TOUCHED" in str(excinfo.value)


def test_a_killed_deps_recreate_converges_the_deps_omn18692(
    executor_seams: dict[str, Any],
    fast_supervisor: None,
    lane_state: dict[str, dict[str, tuple[str, int | None]]],
) -> None:
    """AC3: the agent's next action after a partial recreate is convergence.

    Drives the measured residue -- redpanda and valkey absent, postgres in
    ``created`` -- against a compose CLI that will not exit, so the supervisor
    has to end the command. What must follow is the deps-only ``up -d``, and it
    must precede anything else this process does.
    """
    lane_state["states"] = dict(HALF_RECREATED)
    executor_seams["process"] = _NeverExitingCompose()
    executor = DeployExecutor()
    phases: list[tuple[Phase, PhaseStatus]] = []

    _deps_up(executor, phases)

    convergence = [
        cmd
        for cmd in executor_seams["runs"]
        if "up" in cmd and "--force-recreate" not in cmd
    ]
    assert convergence, "no deps convergence ran after the recreate was ended"
    first = convergence[0]
    for service in DEPS:
        assert service in first
    assert "--force-recreate" not in first, (
        "convergence must START what is absent, never recreate what is running"
    )
    assert lane_state["states"] == RUNNING
    assert (Phase.CORE, PhaseStatus.SUCCESS) in phases

    supervision = executor.recreate_supervision
    assert len(supervision) == 1
    assert supervision[0].outcome is not EnumRecreateOutcome.COMPLETED
    assert supervision[0].mid_recreate_at_decision is True


def test_converge_deps_reads_before_it_acts_omn18692(
    executor_seams: dict[str, Any],
    lane_state: dict[str, dict[str, tuple[str, int | None]]],
) -> None:
    """A healthy lane costs one read and is not mutated.

    This is what lets the agent call convergence unconditionally at startup.
    """
    lane_state["states"] = dict(RUNNING)
    converged, was_down = DeployExecutor().converge_deps(lane=EnumRuntimeLane.DEV)

    assert converged is True
    assert was_down == []
    assert executor_seams["runs"] == []


def test_golden_chain_a_healthy_deps_recreate_reaches_the_receipt_omn18692(
    executor_seams: dict[str, Any],
    fast_supervisor: None,
) -> None:
    """GOLDEN CHAIN: deps recreate -> phase verdict -> terminal payload.

    The chain the 2026-09-10 paired-chain ruling asks for, end to end on the
    happy path: the supervised command completes, the phase passes, and the
    published event carries a supervision record saying so. A deploy whose deps
    phase ran without incident is DISTINGUISHABLE in the receipt from one whose
    deps phase was never reached, which is the property a reader needs.
    """
    executor = DeployExecutor()
    executor.reset_deploy_observations()
    phases: list[tuple[Phase, PhaseStatus]] = []

    _deps_up(executor, phases)

    assert (Phase.CORE, PhaseStatus.SUCCESS) in phases
    assert executor.recreate_supervision[0].outcome is EnumRecreateOutcome.COMPLETED

    payload = _publish(executor)
    assert payload["status"] == "success"
    assert payload["recreate_supervision"][0]["outcome"] == "completed"
    assert payload["recreate_supervision"][0]["phase"] == Phase.CORE.value


def test_error_chain_a_killed_deps_recreate_reaches_the_receipt_omn18692(
    executor_seams: dict[str, Any],
    fast_supervisor: None,
    lane_state: dict[str, dict[str, tuple[str, int | None]]],
) -> None:
    """ERROR CHAIN: the 2026-09-18 shape, driven through to the published event.

    This is the falsifier AC4 names -- a deps-phase kill driven through to the
    runtime's broker-absent state -- expressed where the state is observable:
    the lane loses its broker, the supervisor ends the command, convergence
    brings it back, and the terminal event NAMES the kill. Before this change
    the same sequence published a phase failure with no record of what the
    ceiling had done, and the cause had to be reconstructed from the dockerd
    journal by a third lane 22 minutes later.
    """
    lane_state["states"] = dict(HALF_RECREATED)
    executor_seams["process"] = _NeverExitingCompose()
    executor = DeployExecutor()
    executor.reset_deploy_observations()

    assert "redpanda" not in lane_state["states"], (
        "the fixture must start from the measured residue: no broker at all"
    )

    _deps_up(executor, [])

    # The broker is back, and it is back because THIS process brought it back.
    assert lane_state["states"]["redpanda"] == ("running", None)

    payload = _publish(executor)
    record = payload["recreate_supervision"][0]
    assert record["outcome"] in {
        EnumRecreateOutcome.KILLED_MID_RECREATE_STALLED.value,
        EnumRecreateOutcome.KILLED_HARD_UPPER_BOUND.value,
    }
    assert record["mid_recreate_at_decision"] is True
    # The ceiling this run used is compressed by the fixture; what it was
    # DERIVED FROM is not, and that is the number the flat bound replaced.
    assert record["budget_description"].startswith("1s = compose model")
    assert (
        int(record["budget_description"].split()[4].rstrip("s"))
        > PHASE_TIMEOUTS[Phase.CORE]
    )


def _publish(executor: DeployExecutor) -> dict[str, Any]:
    """Build the terminal payload the agent would publish for this executor."""
    from deploy_agent.job_state import JobState

    now = datetime.now(UTC)
    job = JobState(
        correlation_id=uuid4(),
        command={
            "runtime_lane": EnumRuntimeLane.DEV.value,
            "scope": Scope.CORE.value,
            "git_ref": "origin/dev",
        },
        accepted_at=now,
        completed_at=now,
        phase_results={Phase.CORE: PhaseStatus.SUCCESS},
    )
    payload = build_completion_payload(
        job,
        "0" * 40,
        [],
        services_restarted=list(DEPS),
        container_residue=executor.container_residue,
        sibling_refs=executor.sibling_source_refs,
        recreate_supervision=executor.recreate_supervision,
    )
    # `build_completion_payload` builds this THROUGH ModelRebuildCompleted
    # (OMN-18057), so the payload is validated by construction. Re-validating
    # the dump here would fail on `status`, which is a computed field the model
    # emits and does not accept -- asserting on it below is asserting on the
    # model's own verdict, which is the point.
    assert "status" in payload
    return payload
