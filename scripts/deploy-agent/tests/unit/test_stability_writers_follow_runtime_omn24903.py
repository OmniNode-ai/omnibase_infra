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

import ast
import json
import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from deploy_agent import executor as execution
from deploy_agent.events import BuildSource, EnumRuntimeLane, Phase, Scope
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
OLD_SHA = "a" * 40
WRITER = "projection-delegation-writer"
SENTINEL = ("COMPOSE_ENV_SENTINEL", "set in the agent's process environment")
RUNTIME = "omninode-runtime"
PROJECT = execution.lane_config_for(LANE).compose_project
BROKER_IMAGE = "redpandadata/redpanda:v24.2.7"
NO_WAIT = {"wait_seconds": 0, "poll_seconds": 0, "sleep": lambda seconds: None}


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


def _rendered(service: str, image: str | None) -> dict[str, object]:
    """``config --format json <service>``: the service and the broker it depends on."""
    body: dict[str, object] = {"build": {"context": ".."}, "depends_on": ["redpanda"]}
    if image:
        body["image"] = image
    return {
        "name": PROJECT,
        "services": {service: body, "redpanda": {"image": BROKER_IMAGE}},
    }


@pytest.fixture
def deploy(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A stability deploy with a scripted docker: compose ps/inspect answer from ``state``.

    ``_compose_env`` is the real builder: nothing here stubs the environment the
    writers' compose commands run under.
    """
    state = SimpleNamespace(
        calls=[],
        envs=[],
        returncode=0,
        containers={},
        revisions={},
        images={},
        inspected=[],
        events=[],
    )

    def container(service: str) -> dict[str, object]:
        return state.containers.setdefault(
            service, {"status": "running", "health": "healthy", "restarts": 0}
        )

    def run(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        state.calls.append(argv)
        state.envs.append(kwargs.get("env"))
        out = ""
        if argv[:2] == ["docker", "compose"] and "--images" in argv:
            # What compose prints: the service's image, then its dependencies'.
            out = f"{PROJECT}-{argv[-1]}\n{BROKER_IMAGE}"
        elif argv[:2] == ["docker", "compose"] and "config" in argv:
            out = json.dumps(_rendered(argv[-1], state.images.get(argv[-1])))
        elif argv[:2] == ["docker", "compose"] and "ps" in argv:
            out = f"id-{argv[-1]}"
        elif argv[:2] == ["docker", "inspect"]:
            c = container(argv[-1].removeprefix("id-"))
            out = f"{c['status']}|{c['health']}|{c['restarts']}"
        elif argv[:3] == ["docker", "image", "inspect"]:
            state.inspected.append(argv[-1])
            # Only this lane's built images carry a revision label.
            built = argv[-1].removeprefix(f"{PROJECT}-")
            out = state.revisions.get(built, SHA) if built != argv[-1] else ""
            out = state.revisions.get(argv[-1], out)
        if "up" in argv:
            state.events.append(
                "up:writers"
                if WRITER in argv
                else "up:runtime"
                if RUNTIME in argv
                else "up"
            )
        code = state.returncode if "up" in argv else 0
        return subprocess.CompletedProcess(argv, code, out, "")

    @contextmanager
    def held(purpose: str) -> Iterator[None]:
        state.events.append("lock-enter")
        try:
            yield
        finally:
            state.events.append("lock-exit")

    monkeypatch.setattr(execution, "_run", run)
    monkeypatch.setenv(*SENTINEL)
    monkeypatch.setattr(execution, "_hold_build_context", held)
    state.container = container
    executor = state.executor = DeployExecutor()
    for name in ("_compose_build", "_preflight_runtime_boot", "_deploy_gateway_lane"):
        setattr(executor, name, create_autospec(getattr(executor, name)))
    # The real ``_compose_up``: the roll runs inside it, after the two migration
    # helpers, which record their calls here. Only docker and its container reads
    # are scripted.
    real_up = DeployExecutor._compose_up.__get__(executor)
    executor._compose_up = create_autospec(real_up, side_effect=real_up)
    monkeypatch.setattr(execution, "verify_containers_up", lambda *a, **k: (True, []))
    monkeypatch.setattr(execution, "finish_recreate_renames", lambda lane: None)
    monkeypatch.setattr(execution, "remove_recreate_leftovers", lambda *a: None)
    for name, event in (
        ("_ensure_runtime_migrations_ready", "migrations"),
        ("rerun_changed_oneshots", "oneshots"),
    ):
        monkeypatch.setattr(
            executor, name, lambda event=event, **kwargs: state.events.append(event)
        )
    monkeypatch.setattr(executor, "_supervised_deps_recreate", lambda *a, **k: "")
    monkeypatch.setattr(executor, "observe_deps_convergence", lambda lane: None)
    return state


#: workspace builds stage sibling repos; release builds roll from the dev-merge tag.
BUILD_SOURCES = [BuildSource.WORKSPACE, BuildSource.RELEASE]


def _build(
    state: SimpleNamespace,
    lane: EnumRuntimeLane,
    build_source: BuildSource = BuildSource.WORKSPACE,
) -> tuple[str, ...]:
    built: tuple[str, ...] = _effect().build_stability_lane_writers(
        execution,
        state.executor._compose_build,
        SHA,
        lambda *args: None,
        build_source=build_source,
        lane=lane,
        git_ref="origin/dev",
    )
    return built


def _recreate(state: SimpleNamespace, lane: EnumRuntimeLane) -> tuple[str, ...]:
    rolled: tuple[str, ...] = _effect().roll_writers(
        execution, lane=lane, git_sha=SHA, **NO_WAIT
    )
    return rolled


def _ups(state: SimpleNamespace) -> list[list[str]]:
    return [argv for argv in state.calls if "up" in argv]


def _rebuild(
    state: SimpleNamespace,
    scope: Scope,
    build_source: BuildSource = BuildSource.WORKSPACE,
) -> list[str]:
    return state.executor.rebuild_scope(
        scope,
        [],
        lambda *a: None,
        lane=LANE,
        git_sha=SHA,
        build_source=build_source,
        git_ref="origin/dev",
    )


def test_the_writer_that_refused_the_terminal_is_rolled_by_a_stability_deploy(
    recorded: dict[str, object], deploy: SimpleNamespace
) -> None:
    row = recorded["delegation_events_row"]
    dlq = recorded["malformed_dlq_record"]
    containers = recorded["containers"]
    assert isinstance(row, dict) and isinstance(dlq, dict)
    assert isinstance(containers, dict)
    # The recorded I9 row has no terminal: terminal_ok and the cause are both NULL
    # (the nightly's "terminal unknown"), and its DLQ record is the writer's refusal.
    assert row["terminal_ok"] is None and row["terminal_failure_cause"] is None
    # Both cases the nightly failed read the terminal as 'unknown': I9 wanted
    # 'completed', I10 wanted 'failed'. Read from the job's own FAILED lines.
    assert _terminal_cases(recorded["nightly_failed_lines"]) == {
        "I9": ("completed", "unknown"),
        "I10": ("failed", "unknown"),
    }
    assert row["correlation_id"] == dlq["correlation_id"]
    assert dlq["handler"] == "node_projection_delegation"
    assert "ModelDelegateSkillTerminalProjection" in dlq["failure_reason_lines"][0]
    writer = _service_of_label(containers["writer"]["service_label"])
    assert writer == WRITER

    _build(deploy, LANE)
    _recreate(deploy, LANE)

    built = deploy.executor._compose_build.call_args
    assert built is not None
    assert writer in built.kwargs["services"]
    assert built.kwargs["compose_files"] == (
        execution.lane_config_for(LANE).compose_files
    )
    assert built.kwargs["stage_workspace"] is False
    (up,) = _ups(deploy)
    assert up[up.index("-p") + 1] == execution.lane_config_for(LANE).compose_project
    assert {"--no-deps", "--no-build", "--force-recreate"} <= set(up)
    assert writer in up


_FAILED_CASE = re.compile(
    r"test_case_behaves_as_expected\[(?P<case>I\d+)\] .*?"
    r"terminal: expected '(?P<expected>\w+)', got '(?P<got>\w+)'"
)


def _terminal_cases(lines: object) -> dict[str, tuple[str, str]]:
    """The terminal expectation of each case in the nightly's FAILED lines."""
    assert isinstance(lines, list)
    found = {}
    for line in lines:
        match = _FAILED_CASE.search(str(line))
        if match:
            found[match["case"]] = (match["expected"], match["got"])
    return found


def test_a_fixture_without_the_i10_line_loses_the_i10_case(
    recorded: dict[str, object],
) -> None:
    lines = recorded["nightly_failed_lines"]
    assert isinstance(lines, list) and len(lines) == 2
    kept = [line for line in lines if "[I10]" not in line]
    assert set(_terminal_cases(kept)) == {"I9"}


def test_every_writer_the_stability_overlay_declares_is_rolled() -> None:
    declared = {name for name in _overlay_services() if name.endswith("-writer")}
    assert declared, "positive control: the overlay declares writers"
    assert set(_effect().STABILITY_LANE_WRITER_SERVICES) == declared


@pytest.mark.parametrize("build_source", BUILD_SOURCES)
def test_building_the_writers_recreates_nothing(
    deploy: SimpleNamespace, build_source: BuildSource
) -> None:
    _build(deploy, LANE, build_source)
    assert (
        deploy.executor._compose_build.call_args.kwargs["build_source"] == build_source
    )
    assert deploy.executor._compose_build.call_count == 1
    assert deploy.inspected, "positive control: the build reads the images it wrote"
    assert not _ups(deploy)


def test_a_failed_recreate_fails_the_deploy(deploy: SimpleNamespace) -> None:
    deploy.returncode = 1
    with pytest.raises(RuntimeError, match="writer recreate exited 1"):
        _recreate(deploy, LANE)


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_writer_recreate_exiting_1_fails_the_roll_before_the_runtime_up(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    """The whole roll, not only the helper: the runtime is never recreated."""
    deploy.returncode = 1
    with pytest.raises(RuntimeError, match="writer recreate exited 1"):
        _rebuild(deploy, scope)
    assert len(_ups(deploy)) == 1, "positive control: the writers' up ran and failed"
    _assert_only_the_deps_leg_ran(deploy, scope)
    deploy.executor._deploy_gateway_lane.assert_not_called()


def _assert_only_the_deps_leg_ran(deploy: SimpleNamespace, scope: Scope) -> None:
    """No runtime up: none at all on a RUNTIME roll, the deps leg alone on FULL."""
    assert "up:runtime" not in deploy.events
    phases = [c.args[:2] for c in deploy.executor._compose_up.call_args_list]
    expected = [(Phase.CORE, Scope.CORE)] if scope == Scope.FULL else []
    assert [p for p in phases if p[1] == Scope.CORE] == expected


OTHER_LANES = [lane for lane in EnumRuntimeLane if lane != LANE]


@pytest.mark.parametrize("lane", OTHER_LANES)
def test_another_lane_is_left_untouched(
    deploy: SimpleNamespace, lane: EnumRuntimeLane
) -> None:
    assert len(OTHER_LANES) >= 2, "positive control: lanes without the overlay"
    assert LANE.value in str(execution.lane_config_for(LANE).compose_files)
    assert _build(deploy, lane) == ()
    assert _recreate(deploy, lane) == ()
    deploy.executor._compose_build.assert_not_called()
    assert not deploy.calls


def test_a_writer_the_lane_disables_is_neither_built_nor_recreated(
    deploy: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = execution.lane_config_for(LANE)
    narrowed = config.model_copy(
        update={"disabled_services": config.disabled_services | {WRITER}}
    )
    monkeypatch.setattr(
        execution, "lane_config_for", lambda lane: narrowed if lane == LANE else config
    )
    built = _build(deploy, LANE)
    _recreate(deploy, LANE)
    (up,) = _ups(deploy)
    assert WRITER not in built and WRITER not in up
    assert WRITER not in deploy.executor._compose_build.call_args.kwargs["services"]
    assert "projection-savings-writer" in built, "positive control: others stay"


@pytest.mark.parametrize("build_source", BUILD_SOURCES)
@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_the_writers_are_staged_built_and_recreated_in_the_rolls_order(
    deploy: SimpleNamespace, scope: Scope, build_source: BuildSource
) -> None:
    # OMN-25440: every build, preflight and up lands in one list in call order.
    order: list[str] = deploy.events
    deploy.executor._compose_build.side_effect = lambda *a, **k: order.append(
        "build:writers" if WRITER in k.get("services", ()) else "build:main"
    )
    deploy.executor._preflight_runtime_boot.side_effect = lambda *a, **k: order.append(
        "preflight"
    )
    _rebuild(deploy, scope, build_source)
    last_main = max(i for i, step in enumerate(order) if step == "build:main")
    # The main build stages the workspace at the deploy's sha; the writers' build
    # reuses it (stage_workspace=False), so it must come after a staging build.
    assert last_main < order.index("build:writers")
    main_calls = [
        c
        for c in deploy.executor._compose_build.call_args_list
        if WRITER not in c.kwargs.get("services", ())
    ]
    assert main_calls and all(
        c.kwargs.get("stage_workspace", True) is True for c in main_calls
    )
    writers_call = deploy.executor._compose_build.call_args_list[-1]
    assert writers_call.kwargs["stage_workspace"] is False
    assert writers_call.kwargs["build_source"] == build_source
    _assert_roll_order(order)


#: Nothing running is replaced until the build and the boot preflight passed.
ROLL_ORDER = ["build:writers", "preflight", "up:writers", "up:runtime"]


def _assert_roll_order(order: list[str]) -> None:
    assert [step for step in order if step in ROLL_ORDER] == ROLL_ORDER


def test_a_preflight_recorded_after_the_writers_up_fails_the_order_check() -> None:
    """Negative control for the order check: the order it reads can be wrong."""
    _assert_roll_order(["build:main", *ROLL_ORDER])
    broken = ["build:main", "build:writers", "up:writers", "preflight", "up:runtime"]
    with pytest.raises(AssertionError):
        _assert_roll_order(broken)


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_the_writers_roll_after_the_migrations_and_the_changed_oneshots(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    """New writers start on the migrated schema, as dev's folded writers do.

    The runtime ``_compose_up`` runs ``_ensure_runtime_migrations_ready`` and
    ``rerun_changed_oneshots`` before its up. Both record their calls in the
    fixture, so a roll moved ahead of either fails here, not only a stubbed order.
    """
    _rebuild(deploy, scope)
    steps = ("migrations", "oneshots", "up:writers", "up:runtime")
    assert [e for e in deploy.events if e in steps] == list(steps)


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_the_writers_roll_after_the_build_context_lock_is_released(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    """The lock waits up to 240 s for the reconcile hosts; a roll inside holds it."""
    _rebuild(deploy, scope)
    events = deploy.events
    assert events.count("lock-enter") == events.count("lock-exit") == 1
    assert events.index("lock-exit") < events.index("up:writers")


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_refused_boot_preflight_recreates_no_writer(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    deploy.executor._preflight_runtime_boot.side_effect = RuntimeError("boot refused")
    with pytest.raises(RuntimeError, match="boot refused"):
        _rebuild(deploy, scope)
    assert deploy.executor._compose_build.call_count >= 2, "the writers were built"
    assert _ups(deploy) == []
    deploy.executor._compose_up.assert_not_called()


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_failed_runtime_build_builds_no_writer_and_recreates_none(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    deploy.executor._compose_build.side_effect = RuntimeError("build failed")
    with pytest.raises(RuntimeError, match="build failed"):
        _rebuild(deploy, scope)
    assert deploy.executor._compose_build.call_count == 1
    assert _ups(deploy) == []


BUILDER_ENV = {"WRITER_ROLL_BUILDER_SENTINEL": "returned by the dev-only builder"}


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_the_writer_up_runs_under_the_environment_the_real_runtime_up_gets(
    deploy: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, scope: Scope
) -> None:
    """The builder's environment reaches the runtime up as ``extra_env``; a writer up
    run under ``_compose_env()`` alone would lack it.
    """
    executor = deploy.executor
    monkeypatch.setattr(
        executor, "_build_dev_lane_only_services", lambda *a, **k: dict(BUILDER_ENV)
    )
    _rebuild(deploy, scope)
    ups = [
        (argv, env)
        for argv, env in zip(deploy.calls, deploy.envs, strict=True)
        if "up" in argv
    ]
    writer_ups = [env for argv, env in ups if WRITER in argv]
    runtime_ups = [env for argv, env in ups if RUNTIME in argv]
    assert len(writer_ups) == 1 and runtime_ups, "positive control: both ups ran"
    assert isinstance(writer_ups[0], dict)
    assert writer_ups[0][next(iter(BUILDER_ENV))] == next(iter(BUILDER_ENV.values()))
    assert all(env == writer_ups[0] for env in runtime_ups)
    reads = [
        env
        for argv, env in zip(deploy.calls, deploy.envs, strict=True)
        if argv[:2] == ["docker", "compose"] and "up" not in argv
    ]
    # The builder is replaced here, so the build-time image check does not run:
    # every compose read is the roll's own, under the runtime up's environment.
    assert reads and all(env == writer_ups[0] for env in reads)


def test_the_image_is_read_from_the_services_own_config_entry(
    deploy: SimpleNamespace,
) -> None:
    """``config --images`` lists the broker after the writer; the JSON keys it by name."""
    _recreate(deploy, LANE)
    assert BROKER_IMAGE not in deploy.inspected
    assert set(deploy.inspected) == {
        f"{PROJECT}-{service}" for service in (RUNTIME, *_writers())
    }
    assert not [c for c in deploy.calls if "--images" in c]


def test_an_image_the_compose_file_names_is_the_one_inspected(
    deploy: SimpleNamespace,
) -> None:
    deploy.images[WRITER] = "registry.local/delegation-writer:pinned"
    deploy.revisions["registry.local/delegation-writer:pinned"] = OLD_SHA
    with pytest.raises(RuntimeError, match="not on the runtime's image revision"):
        _recreate(deploy, LANE)
    assert "registry.local/delegation-writer:pinned" in deploy.inspected


def test_healthy_writers_on_the_runtime_image_pass(deploy: SimpleNamespace) -> None:
    assert _recreate(deploy, LANE) == _effect().stability_lane_writer_services(
        execution, LANE
    )


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"status": "restarting", "restarts": 4}, "not stable"),
        ({"status": "running", "restarts": 2}, "not stable"),
        ({"status": "exited"}, "not stable"),
        ({"health": "unhealthy"}, "not stable"),
        ({"health": "starting"}, "not healthy within"),
    ],
    ids=["crash-looping", "restarted", "exited", "unhealthy", "never-healthy"],
)
def test_a_writer_that_is_not_healthy_after_the_recreate_fails_the_deploy(
    deploy: SimpleNamespace, update: dict[str, object], match: str
) -> None:
    deploy.container(WRITER).update(update)
    with pytest.raises(RuntimeError, match=match) as raised:
        _recreate(deploy, LANE)
    assert WRITER in str(raised.value)


def test_a_writer_waits_for_its_healthcheck_before_it_passes(
    deploy: SimpleNamespace,
) -> None:
    deploy.container(WRITER)["health"] = "starting"
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        deploy.container(WRITER)["health"] = "healthy"

    _effect().roll_writers(
        execution, lane=LANE, git_sha=SHA, wait_seconds=60, poll_seconds=5, sleep=sleep
    )
    assert slept == [5]


def _assert_nothing_replaced(deploy: SimpleNamespace) -> None:
    assert _ups(deploy) == [], "zero compose up before the revision check passed"


def test_a_writer_left_on_another_image_than_the_runtime_fails_before_any_up(
    deploy: SimpleNamespace,
) -> None:
    deploy.revisions[WRITER] = OLD_SHA
    with pytest.raises(RuntimeError, match="not on the runtime's image revision") as e:
        _recreate(deploy, LANE)
    assert WRITER in str(e.value) and "projection-savings-writer" not in str(e.value)
    assert deploy.calls, "positive control: the images were read"
    _assert_nothing_replaced(deploy)


def test_the_writers_are_compared_with_the_runtime_image_that_was_read(
    deploy: SimpleNamespace,
) -> None:
    """The runtime's label is read: writers and runtime on the same stale revision fail."""
    deploy.revisions.update({RUNTIME: OLD_SHA, **dict.fromkeys(_writers(), OLD_SHA)})
    with pytest.raises(RuntimeError, match="runtime image revision aaaaaaaaaaaa"):
        _recreate(deploy, LANE)
    _assert_nothing_replaced(deploy)
    read = [c[-1] for c in deploy.calls if c[:3] == ["docker", "image", "inspect"]]
    assert f"{PROJECT}-{RUNTIME}" in read


@pytest.mark.parametrize("revision", ["", "unknown", "<no value>"])
def test_an_image_without_the_deploys_revision_fails_before_any_up(
    deploy: SimpleNamespace, revision: str
) -> None:
    deploy.revisions[WRITER] = revision
    with pytest.raises(RuntimeError, match="not on the runtime's image revision"):
        _recreate(deploy, LANE)
    _assert_nothing_replaced(deploy)


def test_a_recreate_with_no_deploy_sha_proves_nothing_and_touches_nothing(
    deploy: SimpleNamespace,
) -> None:
    with pytest.raises(RuntimeError, match="needs the deploy's git sha"):
        _effect().roll_writers(execution, lane=LANE, git_sha="", **NO_WAIT)
    assert not _ups(deploy)


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_revision_mismatch_is_refused_before_the_preflight_and_the_deps_leg(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    """OMN-25440: read where the writers are built, before anything converges."""
    deploy.revisions[WRITER] = OLD_SHA
    with pytest.raises(RuntimeError, match="not on the runtime's image revision"):
        _rebuild(deploy, scope)
    _assert_nothing_replaced(deploy)
    assert deploy.executor._compose_build.call_count >= 2, "the writers were built"
    deploy.executor._preflight_runtime_boot.assert_not_called()
    deploy.executor._compose_up.assert_not_called()


def _writers() -> tuple[str, ...]:
    return tuple(_effect().STABILITY_LANE_WRITER_SERVICES)


def test_the_overlay_gives_every_writer_a_healthcheck() -> None:
    """A writer with health ``none`` cannot be told from a crashed one after up."""
    assert len(_writers()) == 7, "positive control: the seven writers"
    assert _writers_without_a_healthcheck(_overlay_services()) == []


def test_a_writer_that_drops_its_healthcheck_is_named() -> None:
    services = {n: dict(b) for n, b in _overlay_services().items()}
    services[WRITER].pop("healthcheck")
    assert _writers_without_a_healthcheck(services) == [WRITER]
    services[WRITER]["healthcheck"] = {"disable": True}
    assert _writers_without_a_healthcheck(services) == [WRITER]


def _writers_without_a_healthcheck(
    services: dict[str, dict[str, object]],
) -> list[str]:
    missing = []
    for name in _writers():
        check = services[name].get("healthcheck")
        if not isinstance(check, dict) or check.get("disable") or not check.get("test"):
            missing.append(name)
    return missing


def test_a_writer_with_no_healthcheck_fails_the_roll(deploy: SimpleNamespace) -> None:
    deploy.container(WRITER)["health"] = "none"
    with pytest.raises(RuntimeError, match="not stable") as raised:
        _recreate(deploy, LANE)
    assert WRITER in str(raised.value)


def _runtime_stage() -> str:
    text = (REPO / "docker" / "Dockerfile.runtime").read_text()
    stage = re.search(
        r"^FROM [^\n]* AS runtime\n(?P<body>.*?)(?=^FROM |\Z)", text, re.S | re.M
    )
    assert stage is not None, "the Dockerfile has a runtime stage"
    return stage["body"]


def test_the_runtime_image_label_is_stamped_from_the_build_arg_the_build_passes() -> (
    None
):
    """The label the roll compares is the final stage's, from ``VCS_REF``."""
    stage = _runtime_stage()
    label = _effect().IMAGE_REVISION_LABEL
    assert re.search(r"^ARG VCS_REF$", stage, re.M)
    assert f'{label}="${{VCS_REF}}"' in stage


@pytest.mark.parametrize("build_source", BUILD_SOURCES)
def test_the_build_passes_the_full_deploy_sha_as_the_label_and_cache_args(
    deploy: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, build_source: BuildSource
) -> None:
    argv: list[list[str]] = []

    def run(cmd: list[str], timeout: int, **kwargs: object) -> object:
        argv.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(execution, "_run", run)
    executor = DeployExecutor()
    executor._compose_build(
        Scope.RUNTIME,
        SHA,
        lambda *a: None,
        build_source=build_source,
        runtime_lane=LANE,
        git_ref="origin/dev",
        stage_workspace=False,
    )
    (build,) = [c for c in argv if "build" in c]
    pairs = [build[i + 1] for i, token in enumerate(build) if token == "--build-arg"]
    assert f"GIT_SHA={SHA}" in pairs and f"VCS_REF={SHA}" in pairs


DEV_FOLD_OVERLAY = REPO / "docker" / "docker-compose.dev-tenant-fold.yml"
OVERLAY_TARGETS = ("/etc/onex/skill-overlays", "/etc/onex/skill-overlays-store")
ROOTS_DIR_VAR = "DEV_LANE_SKILL_OVERLAY_ROOTS_DIR"
#: What the deploy agent's process environment holds for every lane it deploys: the
#: operator env sets the roots directory and OMNI_HOME; no env sets the store
#: directory, so both lanes take compose's default for it.
DEPLOY_ENV = {
    ROOTS_DIR_VAR: "/srv/private/overlays",
    "OMNI_HOME": "/srv/registry",
    "ONEX_LANE_DECLARATION_DIR": "/srv/registry/config/lab",
}
OPERATOR_UNIT = (
    REPO / "scripts" / "deploy-agent" / "deploy" / "deploy-agent-dev.service"
)
STORE_DIR_VAR = "DEV_LANE_OVERLAY_STORE_ROOTS_DIR"
_ASSIGNMENT = re.compile(r"^(?:export\s+)?(?P<name>[A-Z_][A-Z0-9_]*)=(?P<value>.*)$")
_VARIABLE = re.compile(r"\$\{(\w+)(?:(:?[-?])([^}]*))?\}")


def _env_file_assignments(text: str) -> dict[str, str]:
    """The ``NAME=value`` lines of a bash-sourced operator env file."""
    found = {}
    for line in text.splitlines():
        match = _ASSIGNMENT.match(line.strip())
        if match:
            found[match["name"]] = match["value"].strip("'\"")
    return found


def _project_directory(lane: EnumRuntimeLane) -> Path:
    """Compose's project directory is the first compose file's directory."""
    return Path(execution.lane_config_for(lane).compose_files[0]).parent


def _interpolate(value: str, env: dict[str, str]) -> str:
    """Compose's interpolation for ``${V}``, ``${V:-d}`` and ``${V:?m}``."""

    def resolve(match: re.Match[str]) -> str:
        name, operator, argument = match.groups()
        if env.get(name):
            return env[name]
        if operator == ":-":
            return argument
        if operator == ":?":
            raise KeyError(f"{name}: {argument}")
        return ""

    return _VARIABLE.sub(resolve, value)


def _overlay_mounts(
    service: dict[str, object], env: dict[str, str], project: Path
) -> list[tuple[str, str, str]]:
    """The (resolved host path, target, mode) of each overlay mount, both syntaxes."""
    mounts = []
    for volume in service.get("volumes") or []:
        if isinstance(volume, dict):
            source, target = (
                str(volume.get("source", "")),
                str(volume.get("target", "")),
            )
            mode = "ro" if volume.get("read_only") else "rw"
        else:
            source, _, rest = _interpolate(str(volume), env).partition(":")
            target, _, mode = rest.partition(":")
            mode = mode or "rw"
        if target in OVERLAY_TARGETS:
            host = os.path.normpath(project / _interpolate(source, env))
            mounts.append((host, target, mode))
    return sorted(mounts)


def _main(services: object) -> dict[str, object]:
    assert isinstance(services, dict)
    main = services["omninode-runtime"]
    assert isinstance(main, dict)
    return main


def test_both_lanes_resolve_their_compose_files_from_one_project_directory() -> None:
    assert _project_directory(EnumRuntimeLane.DEV) == _project_directory(LANE)
    assert _project_directory(LANE) == Path(execution.COMPOSE_FILE).parent, (
        "the relative default of the store directory is read from this directory"
    )


def test_stability_runtime_main_reads_the_same_overlay_roots_as_dev() -> None:
    """RULING 2026-11-17T00:45:37Z: stability takes dev's overlays.

    Without ONEX_SKILL_OVERLAY_ROOTS the stability runtime main refused every
    delegation of the nightly with "no call budget declared".
    """
    dev_main = _main(execution._read_compose_services((str(DEV_FOLD_OVERLAY),)))
    stab_main = _main(_overlay_services())
    dev_env, stab_env = dev_main["environment"], stab_main["environment"]
    assert isinstance(dev_env, dict) and isinstance(stab_env, dict)
    assert dev_env["ONEX_SKILL_OVERLAY_ROOTS"]
    assert (
        stab_env.get("ONEX_SKILL_OVERLAY_ROOTS") == dev_env["ONEX_SKILL_OVERLAY_ROOTS"]
    )
    dev_mounts = _overlay_mounts(
        dev_main, DEPLOY_ENV, _project_directory(EnumRuntimeLane.DEV)
    )
    stab_mounts = _overlay_mounts(stab_main, DEPLOY_ENV, _project_directory(LANE))
    assert sorted(m[1] for m in dev_mounts) == sorted(OVERLAY_TARGETS)
    assert stab_mounts == dev_mounts, "the resolved host directories, not the strings"
    roots = {target: host for host, target, _ in stab_mounts}
    assert roots["/etc/onex/skill-overlays"] == DEPLOY_ENV[ROOTS_DIR_VAR]
    # Line 317's default is relative to the project directory: two levels up from
    # docker/, the registry root's own overlays clone. The resolved path, not a suffix.
    assert roots["/etc/onex/skill-overlays-store"] == os.path.normpath(
        _project_directory(LANE) / ".." / ".." / "overlays" / "overlays"
    )


def test_no_tracked_env_file_sets_the_overlay_store_directory() -> None:
    """Dev sets no store directory, so both lanes take the same relative default."""
    files = [
        *(REPO / "docker").glob("*.env*"),
        *(REPO / "docker").glob("*.txt"),
        REPO / ".env.example",
    ]
    assert len(files) > 3, "positive control: env files were read"
    for path in files:
        assert STORE_DIR_VAR not in _env_file_assignments(path.read_text()), path
    for overlay in (DEV_FOLD_OVERLAY, STABILITY_OVERLAY):
        assert STORE_DIR_VAR in overlay.read_text(), "positive control: it is read"


def test_the_overlay_mount_reader_reads_both_syntaxes_without_failing() -> None:
    project = Path("/p/docker")
    service = {
        "volumes": [
            "relative-only-without-colon",
            "/host/a:/etc/onex/skill-overlays:ro",
            {
                "type": "bind",
                "source": "/host/b",
                "target": "/etc/onex/skill-overlays-store",
                "read_only": True,
            },
        ]
    }
    assert _overlay_mounts(service, {}, project) == [
        ("/host/a", "/etc/onex/skill-overlays", "ro"),
        ("/host/b", "/etc/onex/skill-overlays-store", "ro"),
    ]


# --- OMN-25390 review of fca8c3ea6891 -------------------------------------


def test_every_compose_command_of_the_roll_shares_one_argv_prefix(
    deploy: SimpleNamespace,
) -> None:
    """``up``, ``config`` and ``ps`` all pass ``--profile runtime``.

    The writers sit in the ``runtime`` profile. ``ps`` was the one command that
    lacked it (review 1345-3); a read of a profile the command does not enable
    is not the read ``up`` acts on.
    """
    _recreate(deploy, LANE)
    commands = [c for c in deploy.calls if c[:2] == ["docker", "compose"]]
    prefixes = {}
    for argv in commands:
        sub = next(t for t in argv if t in ("up", "ps", "config"))
        prefixes.setdefault(sub, set()).add(tuple(argv[: argv.index(sub)]))
    assert set(prefixes) == {"up", "ps", "config"}, "positive control: all three ran"
    (shared,) = {p for group in prefixes.values() for p in group}
    assert shared[-2:] == ("--profile", "runtime")
    # Each writer is listed in the wait and again on the final read.
    assert sum("ps" in c for c in commands) == 2 * len(_writers())


def _merged_lane_services() -> dict[str, dict[str, object]]:
    files = [
        str(REPO / "docker" / Path(f).name)
        for f in execution.lane_config_for(LANE).compose_files
    ]
    services = execution._read_compose_services(tuple(files))
    assert isinstance(services, dict)
    return services


def _built_stage(service: dict[str, object], dockerfile_text: str) -> str:
    """The Dockerfile stage compose builds for ``service``: ``target``, else the last."""
    build = service["build"]
    assert isinstance(build, dict)
    assert build["dockerfile"] == "docker/Dockerfile.runtime"
    assert (REPO / "docker" / str(build["context"])).resolve() == REPO
    target = build.get("target")
    if target:
        return str(target)
    return re.findall(r"^FROM [^\n]* AS (\w+)$", dockerfile_text, re.M)[-1]


def test_the_runtime_and_every_writer_build_the_stage_that_carries_the_label() -> None:
    """A writer built from another stage has no revision label: every deploy refused."""
    text = (REPO / "docker" / "Dockerfile.runtime").read_text()
    label = _effect().IMAGE_REVISION_LABEL
    assert f'{label}="${{VCS_REF}}"' in _runtime_stage(), "the runtime stage stamps it"
    services = _merged_lane_services()
    built = (RUNTIME, *_writers())
    assert len(built) == 8, "positive control: the runtime and the seven writers"
    assert {
        name: _built_stage(services[name], text) for name in built
    } == dict.fromkeys(built, "runtime")


def test_a_writer_built_from_another_stage_is_named() -> None:
    text = (REPO / "docker" / "Dockerfile.runtime").read_text()
    build = {"context": "..", "dockerfile": "docker/Dockerfile.runtime"}
    assert _built_stage({"build": build}, text) == "runtime", "positive control"
    assert _built_stage({"build": {**build, "target": "builder"}}, text) == "builder"


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_an_empty_sha_is_refused_at_the_entry_of_rebuild_scope(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    """Before any build, preflight, deps up, lock or docker call."""
    with pytest.raises(RuntimeError, match="needs the deploy's git sha"):
        deploy.executor.rebuild_scope(
            scope, [], lambda *a: None, lane=LANE, git_sha="", git_ref="origin/dev"
        )
    deploy.executor._compose_build.assert_not_called()
    deploy.executor._preflight_runtime_boot.assert_not_called()
    deploy.executor._compose_up.assert_not_called()
    assert deploy.calls == [] and deploy.events == []


def test_an_empty_sha_is_refused_only_where_the_roll_needs_it() -> None:
    roll = _effect().writer_roll
    roll(execution, EnumRuntimeLane.DEV, Scope.RUNTIME, "")
    roll(execution, LANE, Scope.CORE, "")
    roll(execution, LANE, Scope.RUNTIME, SHA)
    with pytest.raises(RuntimeError, match="needs the deploy's git sha"):
        roll(execution, LANE, Scope.RUNTIME, "")


def test_the_only_caller_of_rebuild_scope_passes_the_sha_the_job_pulled() -> None:
    """The agent's job handler is the one production caller; a test per caller."""
    package = Path(execution.__file__).parent
    callers = []
    for path in sorted(package.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "rebuild_scope"
            ):
                keywords = {k.arg: ast.unparse(k.value) for k in node.keywords}
                callers.append((path.name, keywords.get("git_sha")))
    assert callers == [("agent.py", "self._current_git_sha")]
    agent = (package / "agent.py").read_text()
    assert "self._current_git_sha = self.executor.git_pull(" in agent


def test_a_writer_healthy_on_the_first_read_and_unhealthy_on_the_last_fails_the_roll(
    deploy: SimpleNamespace,
) -> None:
    """The wait loop drops a writer once it reads healthy; a final pass re-reads all."""
    first, last = _writers()[0], _writers()[-1]
    base = execution._run
    seen: list[str] = []

    def run(argv: list[str], timeout: int, **kwargs: object) -> object:
        if argv[:2] == ["docker", "inspect"]:
            seen.append(argv[-1])
            if argv[-1] == f"id-{last}":
                deploy.container(first)["health"] = "unhealthy"
        return base(argv, timeout, **kwargs)

    execution._run = run
    try:
        with pytest.raises(RuntimeError, match=first):
            _recreate(deploy, LANE)
    finally:
        execution._run = base
    assert seen.count(f"id-{first}") == 2, "read once in the loop, once at the end"


def _before_verb(argv: list[str], verb: str) -> list[str]:
    return argv[: argv.index(verb)]


def _real_writers_build(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The argv the real ``_compose_build`` runs for the writers' own build call."""
    argv: list[list[str]] = []

    def run(cmd: list[str], timeout: int, **kwargs: object) -> object:
        argv.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(execution, "_run", run)
    # The image check after the build has its own tests; this one reads the build.
    monkeypatch.setattr(_effect(), "_assert_images_at_deploy_sha", lambda *a: None)
    _effect().build_stability_lane_writers(
        execution,
        DeployExecutor()._compose_build,
        SHA,
        lambda *a: None,
        build_source=BuildSource.RELEASE,
        lane=LANE,
        git_ref="origin/dev",
    )
    (build,) = [c for c in argv if "build" in c]
    return build


def test_the_writers_build_runs_under_the_files_and_project_the_roll_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-25440: compose tags a built image ``<project>-<service>``, and the roll
    reads ``<project>-<service>`` of the project ``_compose_prefix`` names, so the
    build's ``-f`` and ``-p`` are the roll's or every stability deploy is refused.
    """
    build = _real_writers_build(monkeypatch)
    assert set(_writers()) <= set(build), "the writers' own call, with its services"
    assert _before_verb(build, "build") == _effect()._compose_prefix(execution, LANE)


def _up_prefixes(deploy: SimpleNamespace) -> tuple[list[str], list[list[str]]]:
    ups = _ups(deploy)
    (writer_up,) = [a for a in ups if WRITER in a]
    runtime_ups = [a for a in ups if RUNTIME in a]
    assert runtime_ups, "positive control: the real _compose_up issued the runtime up"
    return _before_verb(writer_up, "up"), [_before_verb(a, "up") for a in runtime_ups]


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_the_writer_up_has_the_prefix_of_the_runtime_up_the_real_compose_up_issues(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    """OMN-25440: the same ``-f`` files, ``-p`` project and profile as the runtime up."""
    _rebuild(deploy, scope)
    writer, runtime = _up_prefixes(deploy)
    assert all(prefix == writer for prefix in runtime)
    assert "-p" in writer and writer.count("-f") == 2


def test_a_writer_up_that_drops_the_project_is_caught(
    deploy: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: a writer prefix without ``-p`` differs from the runtime's."""
    real = _effect()._compose_prefix

    def without_project(execution_: object, lane: EnumRuntimeLane) -> list[str]:
        argv = real(execution_, lane)
        at = argv.index("-p")
        return argv[:at] + argv[at + 2 :]

    monkeypatch.setattr(_effect(), "_compose_prefix", without_project)
    _rebuild(deploy, Scope.RUNTIME)
    writer, runtime = _up_prefixes(deploy)
    assert all(prefix != writer for prefix in runtime)


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_runtime_up_that_skips_the_handed_over_roll_fails_the_deploy(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    """OMN-25440: ``_compose_up`` runs ``after_migrations`` only in its RUNTIME
    block and returns before it for a request of retired roles only. An up that
    skips the block must not end in a passed deploy with the old writers running.
    """

    def skipping_up(*args: object, **kwargs: object) -> None:
        return None

    deploy.executor._compose_up.side_effect = skipping_up
    with pytest.raises(RuntimeError, match="did not run; the writers were not rolled"):
        _rebuild(deploy, scope)
    assert "up:writers" not in deploy.events
    deploy.executor._deploy_gateway_lane.assert_not_called()


def test_the_roll_handed_to_the_real_runtime_up_runs_and_the_deploy_passes(
    deploy: SimpleNamespace,
) -> None:
    """Positive control for the skip test: the real up runs the step it is handed."""
    _rebuild(deploy, Scope.RUNTIME)
    (call,) = [
        c
        for c in deploy.executor._compose_up.call_args_list
        if c.args[1] == Scope.RUNTIME
    ]
    assert call.kwargs["after_migrations"].ran
    assert "up:writers" in deploy.events


def test_no_job_reaches_rebuild_scope_before_its_git_pull_sets_the_sha() -> None:
    """OMN-25440: ``_current_git_sha`` starts empty and each job resets it to empty;
    the one ``rebuild_scope`` call comes after the ``git_pull`` that sets it, so
    the empty value reaches the roll only through a pull that returned none, which
    ``test_an_empty_sha_is_refused_at_the_entry_of_rebuild_scope`` refuses.
    """
    agent = (Path(execution.__file__).parent / "agent.py").read_text().splitlines()
    resets = [i for i, line in enumerate(agent) if 'self._current_git_sha = ""' in line]
    (pull,) = [
        i
        for i, line in enumerate(agent)
        if "self._current_git_sha = self.executor.git_pull(" in line
    ]
    (rebuild,) = [i for i, line in enumerate(agent) if ".rebuild_scope(" in line]
    assert len(resets) == 2, "the initial value and the per-job reset"
    assert resets[0] < resets[1] < pull < rebuild
