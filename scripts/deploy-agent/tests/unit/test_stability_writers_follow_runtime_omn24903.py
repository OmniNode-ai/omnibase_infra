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
import os
import re
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
OLD_SHA = "a" * 40
WRITER = "projection-delegation-writer"
COMPOSE_ENV = {"COMPOSE_ENV_SENTINEL": "the environment _compose_up runs under"}
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


@pytest.fixture
def deploy(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A stability deploy with a scripted docker: compose ps/inspect answer from ``state``."""
    state = SimpleNamespace(
        calls=[],
        envs=[],
        returncode=0,
        containers={},
        revisions={},
    )

    def container(service: str) -> dict[str, object]:
        return state.containers.setdefault(
            service,
            {
                "status": "running",
                "health": "healthy",
                "restarts": 0,
                "image": f"sha256:{service}",
            },
        )

    def run(
        argv: list[str], timeout: int, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        state.calls.append(argv)
        state.envs.append(kwargs.get("env"))
        out = ""
        if argv[:2] == ["docker", "compose"] and "ps" in argv:
            out = f"id-{argv[-1]}"
        elif argv[:2] == ["docker", "inspect"]:
            c = container(argv[-1].removeprefix("id-"))
            out = f"{c['status']}|{c['health']}|{c['restarts']}|{c['image']}"
        elif argv[:3] == ["docker", "image", "inspect"]:
            image = argv[-1]
            out = state.revisions.get(image, SHA if image.startswith("sha256:") else "")
        code = state.returncode if "up" in argv else 0
        return subprocess.CompletedProcess(argv, code, out, "")

    monkeypatch.setattr(execution, "_run", run)
    monkeypatch.setattr(execution, "_compose_env", lambda extra_env=None: COMPOSE_ENV)
    monkeypatch.setattr(execution, "_hold_build_context", lambda purpose: nullcontext())
    state.container = container
    state.executor = DeployExecutor()
    state.executor._compose_build = Mock()
    state.executor._preflight_runtime_boot = Mock()
    state.executor._compose_up = Mock()
    state.executor._deploy_gateway_lane = Mock()
    return state


def _build(state: SimpleNamespace, lane: EnumRuntimeLane) -> tuple[str, ...]:
    built: tuple[str, ...] = _effect().build_stability_lane_writers(
        execution,
        state.executor._compose_build,
        SHA,
        lambda *args: None,
        build_source=BuildSource.WORKSPACE,
        lane=lane,
        git_ref="origin/dev",
    )
    return built


def _recreate(state: SimpleNamespace, lane: EnumRuntimeLane) -> tuple[str, ...]:
    rolled: tuple[str, ...] = _effect().recreate_stability_lane_writers(
        execution, lane=lane, git_sha=SHA, **NO_WAIT
    )
    return rolled


def _ups(state: SimpleNamespace) -> list[list[str]]:
    return [argv for argv in state.calls if "up" in argv]


def _rebuild(state: SimpleNamespace, scope: Scope) -> list[str]:
    return state.executor.rebuild_scope(
        scope,
        [],
        lambda *a: None,
        lane=LANE,
        git_sha=SHA,
        build_source=BuildSource.WORKSPACE,
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


def test_every_writer_the_stability_overlay_declares_is_rolled() -> None:
    declared = {name for name in _overlay_services() if name.endswith("-writer")}
    assert declared, "positive control: the overlay declares writers"
    assert set(_effect().STABILITY_LANE_WRITER_SERVICES) == declared


def test_the_writers_are_built_before_they_are_recreated(
    deploy: SimpleNamespace,
) -> None:
    order: list[str] = []
    deploy.executor._compose_build.side_effect = lambda *a, **k: order.append("build")
    base = execution._run

    def run(argv: list[str], timeout: int, **kwargs: object) -> object:
        if "up" in argv:
            order.append("up")
        return base(argv, timeout, **kwargs)

    execution._run = run
    try:
        _build(deploy, LANE)
        _recreate(deploy, LANE)
    finally:
        execution._run = base
    assert order == ["build", "up"]


def test_building_the_writers_recreates_nothing(deploy: SimpleNamespace) -> None:
    _build(deploy, LANE)
    assert deploy.executor._compose_build.call_count == 1
    assert not deploy.calls


def test_a_failed_recreate_fails_the_deploy(deploy: SimpleNamespace) -> None:
    deploy.returncode = 1
    with pytest.raises(RuntimeError, match="writer recreate exited 1"):
        _recreate(deploy, LANE)


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


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_the_writers_are_staged_built_and_recreated_in_the_rolls_order(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    order: list[str] = []
    deploy.executor._compose_build.side_effect = lambda *a, **k: order.append(
        "build:writers" if WRITER in k.get("services", ()) else "build:main"
    )
    deploy.executor._preflight_runtime_boot.side_effect = lambda *a, **k: order.append(
        "preflight"
    )
    deploy.executor._compose_up.side_effect = lambda phase, sc, *a, **k: order.append(
        f"up:{sc.value}"
    )
    base = execution._run

    def run(argv: list[str], timeout: int, **kwargs: object) -> object:
        if "up" in argv and WRITER in argv:
            order.append("up:writers")
        return base(argv, timeout, **kwargs)

    execution._run = run
    try:
        _rebuild(deploy, scope)
    finally:
        execution._run = base
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
    # Nothing running is replaced until the build and the boot preflight passed.
    assert (
        order.index("build:writers")
        < order.index("preflight")
        < order.index("up:writers")
        < order.index("up:runtime")
    )
    assert len(_ups(deploy)) == 1


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


def test_the_recreate_and_every_read_use_the_environment_compose_up_uses(
    deploy: SimpleNamespace,
) -> None:
    _recreate(deploy, LANE)
    assert len(deploy.calls) > 3, "positive control: ps, inspect and the up ran"
    assert all(env is COMPOSE_ENV for env in deploy.envs)


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

    _effect().recreate_stability_lane_writers(
        execution, lane=LANE, git_sha=SHA, wait_seconds=60, poll_seconds=5, sleep=sleep
    )
    assert slept == [5]


def test_a_writer_left_on_another_image_than_the_runtime_fails_the_deploy(
    deploy: SimpleNamespace,
) -> None:
    deploy.container(WRITER)["image"] = "sha256:the-old-writer"
    deploy.revisions["sha256:the-old-writer"] = OLD_SHA
    with pytest.raises(RuntimeError, match="not on the runtime's image revision") as e:
        _recreate(deploy, LANE)
    assert WRITER in str(e.value) and "projection-savings-writer" not in str(e.value)


@pytest.mark.parametrize("revision", ["", "unknown"])
def test_a_writer_image_without_the_deploys_revision_fails_the_deploy(
    deploy: SimpleNamespace, revision: str
) -> None:
    deploy.revisions[f"sha256:{WRITER}"] = revision
    with pytest.raises(RuntimeError, match="not on the runtime's image revision"):
        _recreate(deploy, LANE)


def test_a_recreate_with_no_deploy_sha_proves_nothing(deploy: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="needs the deploy's git sha"):
        _effect().recreate_stability_lane_writers(
            execution, lane=LANE, git_sha="", **NO_WAIT
        )


DEV_FOLD_OVERLAY = REPO / "docker" / "docker-compose.dev-tenant-fold.yml"
OVERLAY_TARGETS = ("/etc/onex/skill-overlays", "/etc/onex/skill-overlays-store")
ROOTS_DIR_VAR = "DEV_LANE_SKILL_OVERLAY_ROOTS_DIR"
#: What the deploy agent's process environment holds for every lane it deploys: the
#: operator env sets the roots directory and OMNI_HOME; no env sets the store
#: directory, so both lanes take compose's default for it.
DEPLOY_ENV = {ROOTS_DIR_VAR: "/srv/private/overlays", "OMNI_HOME": "/srv/registry"}
_VARIABLE = re.compile(r"\$\{(\w+)(?:(:?[-?])([^}]*))?\}")


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
    assert roots["/etc/onex/skill-overlays-store"].endswith("/overlays/overlays")


def test_stability_runtime_main_refuses_to_deploy_without_the_roots_directory() -> None:
    """A relative default mounted an empty directory and dropped the budget overlay."""
    stab_main = _main(_overlay_services())
    env = {k: v for k, v in DEPLOY_ENV.items() if k != ROOTS_DIR_VAR}
    with pytest.raises(KeyError, match=ROOTS_DIR_VAR):
        _overlay_mounts(stab_main, env, _project_directory(LANE))
    dev_main = _main(execution._read_compose_services((str(DEV_FOLD_OVERLAY),)))
    assert _overlay_mounts(dev_main, env, _project_directory(EnumRuntimeLane.DEV)), (
        "positive control: dev keeps its default and still resolves"
    )


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
