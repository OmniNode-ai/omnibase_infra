# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24903: a stability runtime deploy also builds and recreates its writers.

The fixture is the recorded case I9 of delegation-regression-nightly run
43826372809: its delegation_events row carries no terminal_ok because the
stability lane's projection-delegation-writer, last built on 2026-11-12, sent
the runtime's delegate-skill terminal to the malformed DLQ. A stability deploy
never named that writer, so it kept the old image while the runtime moved on.
"""

from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy_agent import agent as agent_mod
from deploy_agent import executor as execution
from deploy_agent.events import (
    BuildSource,
    EnumRuntimeLane,
    ModelRebuildRequested,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.job_state import JobStore

from tests.unit.test_agent_prod_stability_digest_guard_wiring_omn15181 import (
    _FakeExecutor,
)

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[4]
LANE = EnumRuntimeLane.STABILITY_TEST
STABILITY_OVERLAY = REPO / "docker" / f"docker-compose.{LANE.value}.yml"
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "stability_nightly_unterminalised_row_omn24903.json"
)
SHA = "c" * 40


def _effect() -> object:
    return execution.load_dev_boot_guard(
        "nodes.node_setup_local_provision_effect.runtime_cost_estimator_build"
    )


def _overlay_services() -> dict[str, dict[str, object]]:
    services = execution._read_compose_services((str(STABILITY_OVERLAY),))
    assert isinstance(services, dict)
    return services


def _service_of_label(label: str) -> str:
    wanted = f"com.omninode.service={label}"
    for name, body in _overlay_services().items():
        if isinstance(body, dict) and wanted in (body.get("labels") or []):
            return name
    raise AssertionError(f"no stability service carries the label {wanted!r}")


@pytest.fixture
def recorded() -> dict[str, object]:
    data = json.loads(FIXTURE.read_text())
    assert isinstance(data, dict)
    return data


@pytest.fixture
def deploy(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(calls=[], returncode=0)

    def run(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        state.calls.append(argv)
        return subprocess.CompletedProcess(argv, state.returncode, "", "")

    monkeypatch.setattr(execution, "_run", run)
    monkeypatch.setattr(execution, "_compose_env", dict)
    monkeypatch.setattr(execution, "_hold_build_context", lambda purpose: nullcontext())
    state.executor = DeployExecutor()
    state.executor._compose_build = Mock()
    state.executor._preflight_runtime_boot = Mock()
    state.executor._compose_up = Mock()
    state.executor._deploy_gateway_lane = Mock()
    return state


def _roll(state: SimpleNamespace, lane: EnumRuntimeLane) -> tuple[str, ...]:
    rolled: tuple[str, ...] = _effect().roll_stability_lane_writers(
        execution,
        state.executor._compose_build,
        SHA,
        lambda *args: None,
        build_source=BuildSource.WORKSPACE,
        lane=lane,
        git_ref="origin/dev",
    )
    return rolled


def _ups(state: SimpleNamespace) -> list[list[str]]:
    return [argv for argv in state.calls if "up" in argv]


def test_the_recorded_row_has_no_terminal_and_its_writer_refused_the_terminal(
    recorded: dict[str, object],
) -> None:
    row = recorded["delegation_events_row"]
    dlq = recorded["malformed_dlq_record"]
    assert isinstance(row, dict) and isinstance(dlq, dict)
    assert row["terminal_ok"] is None
    assert row["correlation_id"] == dlq["correlation_id"]
    assert dlq["handler"] == "node_projection_delegation"
    assert "ModelDelegateSkillTerminalProjection" in dlq["failure_reason_lines"][0]


def test_the_writer_that_refused_the_terminal_is_rolled_by_a_stability_deploy(
    recorded: dict[str, object], deploy: SimpleNamespace
) -> None:
    containers = recorded["containers"]
    assert isinstance(containers, dict)
    writer = _service_of_label(containers["writer"]["service_label"])
    assert writer == "projection-delegation-writer"

    _roll(deploy, EnumRuntimeLane.STABILITY_TEST)

    built = deploy.executor._compose_build.call_args
    assert built is not None
    assert writer in built.kwargs["services"]
    assert built.kwargs["compose_files"] == (
        execution.lane_config_for(EnumRuntimeLane.STABILITY_TEST).compose_files
    )
    assert built.kwargs.get("stage_workspace", True) is True
    (up,) = _ups(deploy)
    assert up[up.index("-p") + 1] == execution.lane_config_for(LANE).compose_project
    assert {"--no-deps", "--no-build", "--force-recreate"} <= set(up)
    assert writer in up


def test_every_writer_the_stability_overlay_declares_is_rolled() -> None:
    declared = {name for name in _overlay_services() if name.endswith("-writer")}
    assert declared, "positive control: the overlay declares writers"
    assert set(_effect().STABILITY_LANE_WRITER_SERVICES) == declared


def test_the_writers_are_built_before_they_are_recreated(
    deploy: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    deploy.executor._compose_build.side_effect = lambda *a, **k: order.append("build")
    recorded_run = execution._run

    def run(argv: list[str], timeout: int, **kwargs: object) -> object:
        order.append("up")
        return recorded_run(argv, timeout, **kwargs)

    monkeypatch.setattr(execution, "_run", run)
    _roll(deploy, EnumRuntimeLane.STABILITY_TEST)
    assert order == ["build", "up"]


def test_a_failed_recreate_fails_the_deploy(deploy: SimpleNamespace) -> None:
    deploy.returncode = 1
    with pytest.raises(RuntimeError, match="writer recreate exited 1"):
        _roll(deploy, EnumRuntimeLane.STABILITY_TEST)


@pytest.mark.parametrize("lane", [EnumRuntimeLane.DEV, EnumRuntimeLane.PROD])
def test_another_lane_is_left_untouched(
    deploy: SimpleNamespace, lane: EnumRuntimeLane
) -> None:
    _roll(deploy, lane)
    deploy.executor._compose_build.assert_not_called()
    assert not deploy.calls


def _executor_roll(state: SimpleNamespace, lane: EnumRuntimeLane) -> tuple[str, ...]:
    rolled: tuple[str, ...] = state.executor.roll_stability_lane_writers(
        SHA,
        lambda *args: None,
        build_source=BuildSource.WORKSPACE,
        lane=lane,
        git_ref="origin/dev",
    )
    return rolled


def test_the_executor_step_builds_then_recreates_the_stability_writers(
    deploy: SimpleNamespace,
) -> None:
    rolled = _executor_roll(deploy, EnumRuntimeLane.STABILITY_TEST)
    assert "projection-delegation-writer" in rolled
    built = deploy.executor._compose_build.call_args
    assert built is not None
    assert set(built.kwargs["services"]) == set(rolled)
    (up,) = _ups(deploy)
    assert set(rolled) <= set(up)


@pytest.mark.parametrize("lane", [EnumRuntimeLane.DEV, EnumRuntimeLane.PROD])
def test_the_executor_step_leaves_another_lane_untouched(
    deploy: SimpleNamespace, lane: EnumRuntimeLane
) -> None:
    assert _executor_roll(deploy, lane) == ()
    deploy.executor._compose_build.assert_not_called()
    assert not deploy.calls


def test_a_stability_runtime_rebuild_itself_still_builds_no_writer(
    deploy: SimpleNamespace,
) -> None:
    """The OMN-18108 build assertion keeps its meaning: the writers are a step
    of their own, so ``rebuild_scope`` names none of them."""
    deploy.executor.rebuild_scope(
        Scope.RUNTIME,
        [],
        lambda *a: None,
        lane=LANE,
        git_sha=SHA,
        build_source=BuildSource.WORKSPACE,
        git_ref="origin/dev",
    )
    named: set[str] = set()
    for call in deploy.executor._compose_build.call_args_list:
        named |= set(call.kwargs.get("services", ()))
    assert named.isdisjoint(_effect().STABILITY_LANE_WRITER_SERVICES)
    assert not [argv for argv in deploy.calls if "--force-recreate" in argv]


class _RollRecordingExecutor(_FakeExecutor):
    def roll_stability_lane_writers(
        self, git_sha: str, on_phase_update: object, **kwargs: object
    ) -> tuple[str, ...]:
        self.calls.append(f"roll:{kwargs['lane']}")
        return ()


def _run_agent_deploy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: EnumRuntimeLane,
    scope: Scope,
) -> list[str]:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="test",
        scope=scope,
        runtime_lane=lane,
    )
    store = JobStore(tmp_path)
    store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path / "agent-state")
    monkeypatch.setattr(
        agent_mod, "capture_rollback_point", lambda executor, lane: object()
    )
    monkeypatch.setattr(agent_mod, "publish_result", lambda payload, config: False)
    agent = agent_mod.DeployAgent(skip_self_update=True)
    agent.job_store = store
    executor = _RollRecordingExecutor(stability_ready_digest=None)
    agent.executor = executor  # type: ignore[assignment]
    agent._run_deploy(cmd)
    return executor.calls


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_stability_deploy_rolls_the_writers_after_the_rebuild_before_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: Scope
) -> None:
    calls = _run_agent_deploy(tmp_path, monkeypatch, LANE, scope)
    assert "rebuild_scope" in calls, calls
    assert f"roll:{LANE}" in calls, calls
    assert calls.index("rebuild_scope") < calls.index(f"roll:{LANE}")
    assert calls.index(f"roll:{LANE}") < calls.index("verify")


@pytest.mark.parametrize(
    ("lane", "scope"),
    [
        (EnumRuntimeLane.DEV, Scope.RUNTIME),
        (EnumRuntimeLane.STABILITY_TEST, Scope.CORE),
    ],
)
def test_no_other_deploy_rolls_the_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: EnumRuntimeLane,
    scope: Scope,
) -> None:
    calls = _run_agent_deploy(tmp_path, monkeypatch, lane, scope)
    assert "rebuild_scope" in calls, "positive control: the deploy ran"
    assert not [call for call in calls if call.startswith("roll:")], calls
