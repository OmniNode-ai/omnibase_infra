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

import pytest
from deploy_agent import executor as execution
from deploy_agent.events import BuildSource, EnumRuntimeLane, Scope
from deploy_agent.executor import DeployExecutor

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
    assert built.kwargs["stage_workspace"] is False
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


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_stability_runtime_deploy_rolls_the_writers_before_the_runtime(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    order: list[str] = []
    deploy.executor._compose_build.side_effect = lambda *a, **k: order.append(
        "build:writers"
        if "projection-delegation-writer" in k.get("services", ())
        else "build"
    )
    deploy.executor._compose_up.side_effect = lambda phase, sc, *a, **k: order.append(
        f"up:{sc.value}"
    )
    deploy.executor.rebuild_scope(
        scope,
        [],
        lambda *a: None,
        lane=LANE,
        git_sha=SHA,
        build_source=BuildSource.WORKSPACE,
        git_ref="origin/dev",
    )
    (up,) = _ups(deploy)
    assert "projection-delegation-writer" in up
    assert "build:writers" in order
    assert order.index("build:writers") < order.index("up:runtime")
