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
        if argv[:2] == ["docker", "compose"] and "config" in argv:
            out = f"image-of-{argv[-1]}"
        elif argv[:2] == ["docker", "compose"] and "ps" in argv:
            out = f"id-{argv[-1]}"
        elif argv[:2] == ["docker", "inspect"]:
            c = container(argv[-1].removeprefix("id-"))
            out = f"{c['status']}|{c['health']}|{c['restarts']}"
        elif argv[:3] == ["docker", "image", "inspect"]:
            out = state.revisions.get(argv[-1].removeprefix("image-of-"), SHA)
        code = state.returncode if "up" in argv else 0
        return subprocess.CompletedProcess(argv, code, out, "")

    monkeypatch.setattr(execution, "_run", run)
    monkeypatch.setenv(*SENTINEL)
    monkeypatch.setattr(execution, "_hold_build_context", lambda purpose: nullcontext())
    state.container = container
    state.executor = DeployExecutor()
    for name in (
        "_compose_build",
        "_preflight_runtime_boot",
        "_compose_up",
        "_deploy_gateway_lane",
    ):
        setattr(state.executor, name, create_autospec(getattr(state.executor, name)))
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
    rolled: tuple[str, ...] = _effect().recreate_stability_lane_writers(
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
    # 'completed', I10 wanted 'failed'. Only I9's row was readable to record.
    cases = {c["case"]: c for c in recorded["nightly_cases"]}
    assert set(cases) == {"I9", "I10"}
    assert (cases["I9"]["expected"], cases["I9"]["got"]) == ("completed", "unknown")
    assert (cases["I10"]["expected"], cases["I10"]["got"]) == ("failed", "unknown")
    assert {c["check"] for c in cases.values()} == {"terminal"}
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


@pytest.mark.parametrize("build_source", BUILD_SOURCES)
def test_building_the_writers_recreates_nothing(
    deploy: SimpleNamespace, build_source: BuildSource
) -> None:
    _build(deploy, LANE, build_source)
    assert (
        deploy.executor._compose_build.call_args.kwargs["build_source"] == build_source
    )
    assert deploy.executor._compose_build.call_count == 1
    assert not deploy.calls


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
    phases = [c.args[0] for c in deploy.executor._compose_up.call_args_list]
    assert Phase.RUNTIME not in phases
    deploy.executor._deploy_gateway_lane.assert_not_called()


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
        elif argv[:2] == ["docker", "inspect"] and "health" not in order:
            order.append("health")
        return base(argv, timeout, **kwargs)

    execution._run = run
    try:
        _rebuild(deploy, scope, build_source)
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
    assert writers_call.kwargs["build_source"] == build_source
    # Nothing running is replaced until the build and the boot preflight passed and,
    # on a FULL roll, the deps leg converged: a writer started before the broker and
    # the database are up would fail its health wait and the deploy with it.
    expected = ["build:writers", "preflight"]
    expected += ["up:core"] if scope == Scope.FULL else []
    expected += ["up:writers", "health", "up:runtime"]
    assert [step for step in order if step in expected] == expected
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
    deploy: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither side is stubbed: ``_compose_up`` runs its steps under ``_compose_env``."""
    monkeypatch.delenv("OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN", False)
    # What _compose_up hands compose for a stack up, through the real step runner.
    execution.load_dev_boot_guard().run_compose_up_steps(
        [["docker", "compose", "up", "-d"]],
        10,
        rolling=False,
        lane=LANE,
        execution=execution,
    )
    (stack_env,) = deploy.envs
    deploy.envs.clear()
    deploy.calls.clear()
    _recreate(deploy, LANE)
    assert len(deploy.calls) > 3, "positive control: ps, inspect and the up ran"
    assert stack_env == execution._compose_env()
    assert all(env == stack_env for env in deploy.envs)
    # A bare environment would lack both: the sentinel is the process's, the DSN is
    # the builder's own default and is in no process environment here.
    assert stack_env[SENTINEL[0]] == SENTINEL[1]
    assert "OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN" in stack_env
    assert "OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN" not in os.environ


def test_a_writer_up_given_a_bare_environment_is_caught(
    deploy: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: a bare ``os.environ`` is not the builder's environment."""
    monkeypatch.delenv("OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN", False)
    builders = execution._compose_env()
    monkeypatch.setattr(
        execution, "_compose_env", lambda extra_env=None: dict(os.environ)
    )
    _recreate(deploy, LANE)
    assert deploy.envs and all(env != builders for env in deploy.envs)


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
    assert f"image-of-{RUNTIME}" in read


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
        _effect().recreate_stability_lane_writers(
            execution, lane=LANE, git_sha="", **NO_WAIT
        )
    assert not _ups(deploy)


@pytest.mark.parametrize("scope", [Scope.RUNTIME, Scope.FULL])
def test_a_revision_mismatch_in_the_roll_runs_no_writer_up_and_no_runtime_up(
    deploy: SimpleNamespace, scope: Scope
) -> None:
    deploy.revisions[WRITER] = OLD_SHA
    with pytest.raises(RuntimeError, match="not on the runtime's image revision"):
        _rebuild(deploy, scope)
    _assert_nothing_replaced(deploy)
    phases = [c.args[0] for c in deploy.executor._compose_up.call_args_list]
    assert Phase.RUNTIME not in phases


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
    assert len(SHA) == 40


DEV_FOLD_OVERLAY = REPO / "docker" / "docker-compose.dev-tenant-fold.yml"
OVERLAY_TARGETS = ("/etc/onex/skill-overlays", "/etc/onex/skill-overlays-store")
ROOTS_DIR_VAR = "DEV_LANE_SKILL_OVERLAY_ROOTS_DIR"
#: What the deploy agent's process environment holds for every lane it deploys: the
#: operator env sets the roots directory and OMNI_HOME; no env sets the store
#: directory, so both lanes take compose's default for it.
DEPLOY_ENV = {ROOTS_DIR_VAR: "/srv/private/overlays", "OMNI_HOME": "/srv/registry"}
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


def test_the_deploy_environment_comes_from_the_operator_env_file(
    tmp_path: Path,
) -> None:
    """The agent's environment is its unit's env file; the deploy refuses without the var.

    The file is the host's (``DEPLOY_AGENT_ENV_FILE`` of the unit tracked here), so a
    hosted test reads a file written to match it: the roots directory and OMNI_HOME
    set, the store directory not set.
    """
    unit = OPERATOR_UNIT.read_text()
    assert re.search(r"^Environment=DEPLOY_AGENT_ENV_FILE=\S+$", unit, re.M)
    operator_env = tmp_path / ".env"
    operator_env.write_text(
        f"export {ROOTS_DIR_VAR}='/srv/private/overlays'\nOMNI_HOME=/srv/registry\n"
    )
    env = _env_file_assignments(operator_env.read_text())
    assert env == DEPLOY_ENV and STORE_DIR_VAR not in env
    stab_main = _main(_overlay_services())
    assert _overlay_mounts(stab_main, env, _project_directory(LANE))
    operator_env.write_text("OMNI_HOME=/srv/registry\n")
    absent = _env_file_assignments(operator_env.read_text())
    with pytest.raises(KeyError, match=ROOTS_DIR_VAR):
        _overlay_mounts(stab_main, absent, _project_directory(LANE))


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
