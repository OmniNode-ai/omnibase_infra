# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's runtime binds the commitment ledger source (OMN-24218).

``node_commitment_check_scheduled_effect`` and ``node_work_state_projection_scheduled_effect``
read the ledger through commands named by variables; unset, every tick failed with
"the commitment ledger source is not bound in the runtime". The lane overlay binds those
variables and mounts the lane declaration read-only so the ledger read resolves its broker
address. Primary dev folds the effects role into main, so the fold overlay carries the same
binding on ``omninode-runtime``.

Those readers live in omnimarket, which this repository does not install, so these checks
cover the shipped compose configuration: the committed entries and the value compose renders
from them. The running container is read on the lab after the deploy agent converges.
"""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import click
import pytest

from omnibase_infra.cli.cli_ledger import ledger_query
from tests.ci._compose_render_omn21099 import (
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    DOCKER_DIR,
    REPO_ROOT,
    compose_config,
    docker_compose_available,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

CONFIG_ROOT_PREFIX = "env ONEX_WORKSPACE_CONFIG_ROOT=/etc/onex/workspace-config"
LEDGER_ENV = {
    "ONEX_LEDGER_BUS_READ_COMMAND": (
        f"{CONFIG_ROOT_PREFIX} OMNIBASE_PATH=/app python -m omnibase_core.cli.cli_commands ledger query --bus-lane dev --stdin"
    ),
    "ONEX_LEDGER_BUS_APPEND_COMMAND": (
        f"{CONFIG_ROOT_PREFIX} python -m omnimarket.work_ledger_bus.cli append "
        "--bus-lane dev --omnibase-path /app"
    ),
    "ONEX_LEDGER_HOST_NAME": "${DEV_LANE_LEDGER_HOST_NAME:-dev-runtime}",
}
LEDGER_MOUNT_TARGET = "/etc/onex/workspace-config/config/lab"

REQUIRED_SOURCE_PREFIX = "${ONEX_LANE_DECLARATION_DIR:?"
FOLD_OVERLAY = "docker-compose.dev-tenant-fold.yml"
RUNTIME_BUILD_DIR = REPO_ROOT / "scripts" / "runtime_build"
COMPOSE_FILES_LIB = RUNTIME_BUILD_DIR / "compose_files.sh"


def require_compose(available: bool, in_ci: bool) -> None:
    """The render tests below need the compose CLI: skipped off CI, a failure on CI.

    The tests below render the stack, so a CI runner without the CLI must not
    quietly leave the committed-entry string pins as the only evidence.
    """
    if available:
        return
    if in_ci:
        pytest.fail(
            "docker compose CLI absent on CI: the render tests "
            "may not be skipped there (OMN-24854)"
        )
    pytest.skip("docker compose CLI absent; the committed-entry checks still hold")


@pytest.fixture
def compose_cli() -> None:
    # GitHub Actions sets CI and GITHUB_ACTIONS for every step; test_no_workflow_clears_the_ci_variables_the_compose_fixture_reads pins that no workflow clears either.
    in_ci = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
    require_compose(docker_compose_available(), in_ci)


needs_compose = pytest.mark.usefixtures("compose_cli")


def _services(filename: str) -> dict[str, dict[str, object]]:
    services = load_compose(filename)["services"]
    assert isinstance(services, dict)
    return services


def _environment_of(service: dict[str, object]) -> dict[str, object]:
    env = service.get("environment", {})
    return env if isinstance(env, dict) else {}


def _mounts_of(service: dict[str, object]) -> list[str]:
    volumes = service.get("volumes", [])
    assert isinstance(volumes, list)
    return [str(v) for v in volumes if LEDGER_MOUNT_TARGET in str(v)]


def test_runtime_effects_binds_the_ledger_source() -> None:
    env = _environment_of(_services(DEV_LANE_OVERLAY)["runtime-effects"])
    for name, value in LEDGER_ENV.items():
        assert env[name] == value


def test_runtime_effects_mounts_lane_declaration_read_only() -> None:
    mounts = _mounts_of(_services(DEV_LANE_OVERLAY)["runtime-effects"])
    assert len(mounts) == 1
    assert mounts[0].endswith(f":{LEDGER_MOUNT_TARGET}:ro")


def test_the_config_root_is_not_a_container_wide_variable() -> None:
    # The root rides inside the two commands; no other reader in the container sees it.
    for filename, name in (
        (DEV_LANE_OVERLAY, "runtime-effects"),
        ("docker-compose.dev-tenant-fold.yml", "omninode-runtime"),
    ):
        env = _environment_of(_services(filename)[name])
        assert "ONEX_WORKSPACE_CONFIG_ROOT" not in env
        assert "OMNIBASE_PATH" not in env


def test_no_other_dev_lane_service_gains_the_ledger_variables() -> None:
    holders = sorted(
        name
        for name, service in _services(DEV_LANE_OVERLAY).items()
        if isinstance(service, dict)
        and "ONEX_LEDGER_BUS_READ_COMMAND" in _environment_of(service)
    )
    assert holders == ["runtime-effects"]


def test_folded_main_binds_the_same_ledger_source() -> None:
    service = _services("docker-compose.dev-tenant-fold.yml")["omninode-runtime"]
    env = _environment_of(service)
    for name, value in LEDGER_ENV.items():
        assert env[name] == value
    mounts = _mounts_of(service)
    assert len(mounts) == 1
    assert mounts[0].endswith(f":{LEDGER_MOUNT_TARGET}:ro")


def test_the_committed_mount_source_requires_the_declaration_dir() -> None:
    # OMN-24854: the deploy agent points ONEX_LANE_OVERLAY_DIR at a rendered copy
    # under its state directory, so the declaration folder travels under its own
    # name, required, never a nested default (the repository bans those).
    for filename, name in (
        (DEV_LANE_OVERLAY, "runtime-effects"),
        ("docker-compose.dev-tenant-fold.yml", "omninode-runtime"),
    ):
        mounts = _mounts_of(_services(filename)[name])
        assert len(mounts) == 1, filename
        assert mounts[0].startswith(REQUIRED_SOURCE_PREFIX), filename
        assert mounts[0].endswith(f":{LEDGER_MOUNT_TARGET}:ro"), filename


def test_compose_absence_is_a_failure_on_ci_and_a_skip_elsewhere() -> None:
    require_compose(available=True, in_ci=True)
    require_compose(available=True, in_ci=False)
    with pytest.raises(pytest.fail.Exception, match="may not be skipped"):
        require_compose(available=False, in_ci=True)
    with pytest.raises(pytest.skip.Exception):
        require_compose(available=False, in_ci=False)


def test_no_compose_file_mounts_the_overlay_dir_parent() -> None:
    # OMN-24854: the overlay directory is the deploy agent's rendered copy, so its
    # parent is no lane declaration, so no compose file may mount it.
    offenders = sorted(
        f"{path.relative_to(DOCKER_DIR)}:{number}"
        for path in sorted((*DOCKER_DIR.rglob("*.yml"), *DOCKER_DIR.rglob("*.yaml")))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\$\{ONEX_LANE_OVERLAY_DIR\}/\.\.", line)
        and not line.strip().startswith("#")
    )
    assert offenders == []


def _rendered_declaration_mount(stack: tuple[str, ...], name: str) -> dict[str, Any]:
    services = render_stack(
        stack,
        {
            "ONEX_LANE_OVERLAY_DIR": "/srv/state/rendered-lane-overlays",
            "ONEX_LANE_DECLARATION_DIR": "/srv/workspace/config/lab",
        },
    )
    volumes = services[name]["volumes"]
    mounts = [v for v in volumes if v["target"] == LEDGER_MOUNT_TARGET]
    assert len(mounts) == 1
    return mounts[0]


@needs_compose
def test_runtime_effects_mount_follows_the_declaration_dir_not_the_rendered_overlay_dir() -> (
    None
):
    # OMN-24854: the deploy agent points ONEX_LANE_OVERLAY_DIR at its rendered
    # copy, so the mount source must come from ONEX_LANE_DECLARATION_DIR; the
    # rendered state directory holds no lane declaration to resolve the broker
    # address from.
    mount = _rendered_declaration_mount(DEV_LANE_STACK, "runtime-effects")

    assert posixpath.normpath(mount["source"]) == "/srv/workspace/config/lab"
    assert mount["read_only"] is True


@needs_compose
def test_folded_main_mount_follows_the_declaration_dir_not_the_rendered_overlay_dir() -> (
    None
):
    stack = (*DEV_LANE_STACK, "docker-compose.dev-tenant-fold.yml")

    mount = _rendered_declaration_mount(stack, "omninode-runtime")

    assert posixpath.normpath(mount["source"]) == "/srv/workspace/config/lab"
    assert mount["read_only"] is True


@needs_compose
@pytest.mark.parametrize("stack", [DEV_LANE_STACK, (*DEV_LANE_STACK, FOLD_OVERLAY)])
def test_a_render_with_only_the_overlay_dir_set_is_refused_naming_the_declaration_dir(
    stack: tuple[str, ...],
) -> None:
    # OMN-24854: the overlay directory's parent is no longer the mount source, so a caller
    # that sets only ONEX_LANE_OVERLAY_DIR (every caller before this change) must be refused
    # by compose, by name, never rendered with a guessed folder.
    refused = compose_config(
        stack,
        {"ONEX_LANE_OVERLAY_DIR": "/srv/state/rendered-lane-overlays"},
        omit=("ONEX_LANE_DECLARATION_DIR",),
    )

    assert refused.returncode != 0
    assert "ONEX_LANE_DECLARATION_DIR must name the config/lab folder" in refused.stderr


def _dev_lane_shell(
    tmp_path: Path, body: str, **env: str
) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in bash after sourcing the compose file helpers, with only ``env`` set."""
    docker_dir = tmp_path / "deploy" / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["bash", "-c", f'source "{COMPOSE_FILES_LIB}"\n{body}'],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], **env},
    )


@pytest.fixture
def lab_config(tmp_path: Path) -> dict[str, Path]:
    """A config root whose config/lab holds the lane declaration, and a rendered overlay copy."""
    config_lab = tmp_path / "config" / "lab"
    lane_overlays = config_lab / "lane-overlays"
    rendered = tmp_path / "state" / "rendered-lane-overlays"
    for folder in (lane_overlays, rendered):
        folder.mkdir(parents=True)
        (folder / "omnibase-infra.bifrost.yaml").write_text("lane: dev\n")
    (config_lab / "ci_bus_lanes.yaml").write_text("lanes: {}\n")
    return {
        "root": tmp_path,
        "config_lab": config_lab,
        "lane_overlays": lane_overlays,
        "rendered": rendered,
        "state": tmp_path / "state",
    }


def _resolved_declaration_dir(tmp_path: Path, entry: str, **env: str) -> str:
    """What ``ONEX_LANE_DECLARATION_DIR`` holds after the helper's ``entry`` runs."""
    result = _dev_lane_shell(
        tmp_path,
        "declare -a args\n"
        f"{entry.format(deploy=tmp_path / 'deploy')}\n"
        'printf "%s" "${ONEX_LANE_DECLARATION_DIR-UNSET}"',
        **env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


ENTRIES = [
    'resolve_compose_file_args args "{deploy}" omnibase-infra',
    "resolve_lane_overlay_dir omnibase-infra",
]


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_compose_file_helper_takes_the_declaration_dir_from_the_workspace_config_root(
    tmp_path: Path, lab_config: dict[str, Path], entry: str
) -> None:
    # The deploy agent's rendered overlay directory is the only overlay variable here, as in a
    # shell that applied a refresh from before ONEX_LANE_DECLARATION_DIR existed.
    resolved = _resolved_declaration_dir(
        tmp_path,
        entry,
        ONEX_LANE_OVERLAY_DIR=str(lab_config["rendered"]),
        ONEX_WORKSPACE_CONFIG_ROOT=str(lab_config["root"]),
    )

    assert resolved == str(lab_config["config_lab"].resolve())


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_compose_file_helper_keeps_a_pinned_declaration_dir(
    tmp_path: Path, lab_config: dict[str, Path], entry: str
) -> None:
    pinned = lab_config["config_lab"]
    resolved = _resolved_declaration_dir(
        tmp_path,
        entry,
        ONEX_LANE_OVERLAY_DIR=str(lab_config["rendered"]),
        ONEX_LANE_DECLARATION_DIR=str(pinned),
    )

    assert resolved == str(pinned.resolve())


@pytest.mark.parametrize("entry", ENTRIES)
def test_control_the_compose_file_helper_leaves_the_variable_unset_for_a_rendered_overlay_dir_alone(
    tmp_path: Path, lab_config: dict[str, Path], entry: str
) -> None:
    # No pinned folder and no config root: the rendered directory's parent is the agent's
    # state directory, which holds no declaration, so compose must refuse by name.
    resolved = _resolved_declaration_dir(
        tmp_path, entry, ONEX_LANE_OVERLAY_DIR=str(lab_config["rendered"])
    )

    assert resolved == "UNSET"


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_compose_file_helper_takes_the_declaration_dir_from_the_real_overlay_dir_alone(
    tmp_path: Path, lab_config: dict[str, Path], entry: str
) -> None:
    # OMN-25202: a caller that holds only the real ONEX_LANE_OVERLAY_DIR (no pin, no config
    # root) gets the folder above it, which holds ci_bus_lanes.yaml, like the Python derivation.
    resolved = _resolved_declaration_dir(
        tmp_path, entry, ONEX_LANE_OVERLAY_DIR=str(lab_config["lane_overlays"])
    )

    assert resolved == str(lab_config["config_lab"].resolve())


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_real_overlay_dir_wins_over_a_config_root_that_names_another_folder(
    tmp_path: Path, lab_config: dict[str, Path], entry: str
) -> None:
    other = tmp_path / "other-root" / "config" / "lab"
    other.mkdir(parents=True)
    (other / "ci_bus_lanes.yaml").write_text("lanes: {}\n")
    resolved = _resolved_declaration_dir(
        tmp_path,
        entry,
        ONEX_LANE_OVERLAY_DIR=str(lab_config["lane_overlays"]),
        ONEX_WORKSPACE_CONFIG_ROOT=str(tmp_path / "other-root"),
    )

    assert resolved == str(lab_config["config_lab"].resolve())


def _stale_pin_env(lab_config: dict[str, Path]) -> dict[str, str]:
    # A pinned folder that holds no ci_bus_lanes.yaml, beside a real overlay dir and config
    # root that could each supply one: a stale pin is never replaced by them.
    return {
        "ONEX_LANE_OVERLAY_DIR": str(lab_config["lane_overlays"]),
        "ONEX_LANE_DECLARATION_DIR": str(lab_config["state"]),
        "ONEX_WORKSPACE_CONFIG_ROOT": str(lab_config["root"]),
    }


def test_the_strict_form_stops_before_any_compose_call_on_a_stale_pinned_declaration_dir(
    tmp_path: Path, lab_config: dict[str, Path]
) -> None:
    result = _dev_lane_shell(
        tmp_path,
        "resolve_lane_overlay_dir omnibase-infra || exit $?\n"
        "echo COMPOSE-WOULD-RUN\n"
        'printf "%s" "${ONEX_LANE_DECLARATION_DIR-UNSET}"',
        **_stale_pin_env(lab_config),
    )

    assert result.returncode == 1
    assert "COMPOSE-WOULD-RUN" not in result.stdout
    assert str(lab_config["state"]) in result.stderr
    assert "ONEX_LANE_DECLARATION_DIR" in result.stderr


def test_the_export_form_names_a_stale_pinned_declaration_dir_and_leaves_it_unset(
    tmp_path: Path, lab_config: dict[str, Path]
) -> None:
    # The export form runs before every lane compose call and never aborts its caller:
    # compose itself refuses at the mount, which names the variable.
    result = _dev_lane_shell(
        tmp_path,
        "declare -a args\n"
        f'resolve_compose_file_args args "{tmp_path / "deploy"}" omnibase-infra\n'
        'printf "%s" "${ONEX_LANE_DECLARATION_DIR-UNSET}"',
        **_stale_pin_env(lab_config),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "UNSET"
    assert str(lab_config["state"]) in result.stderr


def test_control_the_strict_form_succeeds_when_the_pinned_declaration_dir_holds_the_file(
    tmp_path: Path, lab_config: dict[str, Path]
) -> None:
    env = _stale_pin_env(lab_config)
    env["ONEX_LANE_DECLARATION_DIR"] = str(lab_config["config_lab"])
    result = _dev_lane_shell(
        tmp_path,
        "resolve_lane_overlay_dir omnibase-infra || exit $?\n"
        'printf "%s" "${ONEX_LANE_DECLARATION_DIR-UNSET}"',
        **env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == str(lab_config["config_lab"].resolve())


# OMN-25202: every file outside tests and docs that spells a dev-lane compose file as a compose
# input, mapped to how the file's environment gets ONEX_LANE_DECLARATION_DIR before compose
# reads it. A file that is not listed fails the scan until it is classified, so a new caller
# cannot pass by never being looked at.
LAYERING_FILES = ("docker-compose.dev-lane.yml", "docker-compose.dev-tenant-fold.yml")
LAYERING_LINE = re.compile(
    r"(?:(?:^|\s)(?:-f|--compose-file|--compose)[ =]\"?\S*"
    r"|^\s*[A-Za-z_]\w*\s*[:=]\s*f?\"?\S*)"
    r"docker-compose\.(?:dev-lane|dev-tenant-fold)\.yml"
)
HELPER_CALL = re.compile(
    r"^\s*(?:resolve_lane_overlay_dir|export_lane_overlay_dir)\b", re.M
)
CALLERS = {
    # The helper library: the layering sits in resolve_compose_file_args, after the export call.
    "scripts/runtime_build/compose_files.sh": "helper-before-layering",
    # The pre-PR slot: resolve_lane_overlay_dir (strict) runs before its compose array exists.
    "scripts/runtime_build/prepr_verify_lane.sh": "helper-before-layering",
    # The warm refresh names both files only to the preflight, which reads this shell's
    # environment: the refresh_exports whose output is sourced comes before the preflight call.
    "scripts/runtime_build/refresh_dev_lane.sh": "exports-before-layering",
    # The agent: every job's compose environment is _compose_env, which takes the render.
    "scripts/deploy-agent/deploy_agent/executor.py": "agent-compose-env",
    # Read as YAML (declared_consumer_groups.py imports no subprocess): no compose call.
    ".github/workflows/reusable-runtime-boot.yml": "reads-yaml-only",
    ".github/workflows/runtime-rebuild-trigger.yml": "reads-yaml-only",
}
SCAN_ROOTS = ("scripts", "src", ".github/workflows")
SCAN_SUFFIXES = {".sh", ".py", ".yml", ".yaml"}


def _layering_files() -> dict[str, list[tuple[int, str]]]:
    """Each file under the scan roots with its non-comment lines that layer a dev-lane file."""
    found: dict[str, list[tuple[int, str]]] = {}
    for root in SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if path.suffix not in SCAN_SUFFIXES or "tests" in path.parts:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            hits = [
                (number, line)
                for number, line in enumerate(lines, 1)
                if not line.lstrip().startswith("#") and LAYERING_LINE.search(line)
            ]
            if hits:
                found[str(path.relative_to(REPO_ROOT))] = hits
    return found


def _check_caller(kind: str, text: str, hits: list[tuple[int, str]]) -> None:
    """Raise ``AssertionError`` when ``text`` reaches compose before its variable is set."""
    first = min(number for number, _ in hits)
    before = "\n".join(text.splitlines()[: first - 1])
    if kind == "helper-before-layering":
        assert HELPER_CALL.search(before), "no overlay helper call before the layering"
    elif kind == "exports-before-layering":
        # The layering lines sit in a function definition; the call is what reaches compose.
        call = re.search(r"^\s*if ! run_preflight_required_env\b", text, re.M)
        assert call is not None
        sourced = text.find("source <(printf '%s' \"${LANE_ENV_EXPORTS}\")")
        assert 0 <= sourced < call.start(), "exports not sourced before the preflight"
        assert "mounts_lane_declaration=True" in text[:sourced], (
            "no dev refresh_exports"
        )
    elif kind == "agent-compose-env":
        body = re.search(r"^def _compose_env\(.*?^def ", text, re.M | re.S)
        assert body is not None
        assert "_llm_endpoint_env_overlay(env)" in body.group(0)
    else:
        raise AssertionError(f"unknown caller kind {kind}")


def test_every_file_that_layers_a_dev_lane_compose_file_is_classified() -> None:
    found = _layering_files()

    assert sorted(found) == sorted(CALLERS)
    assert {
        name
        for hits in found.values()
        for _, line in hits
        for name in LAYERING_FILES
        if name in line
    } == set(LAYERING_FILES)


@pytest.mark.parametrize(
    "path", [p for p, kind in CALLERS.items() if kind != "reads-yaml-only"]
)
def test_every_shell_or_agent_caller_sets_the_declaration_dir_before_its_compose_call(
    path: str,
) -> None:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")

    _check_caller(CALLERS[path], text, _layering_files()[path])


@pytest.mark.parametrize(
    "path", [p for p, kind in CALLERS.items() if kind != "reads-yaml-only"]
)
def test_control_a_caller_without_its_helper_call_fails_the_scan(path: str) -> None:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    hits = _layering_files()[path]
    removed = {
        "helper-before-layering": HELPER_CALL.sub("true # removed", text),
        "exports-before-layering": text.replace("mounts_lane_declaration=True", "x"),
        "agent-compose-env": text.replace("_llm_endpoint_env_overlay(env)", "{}"),
    }[CALLERS[path]]
    assert removed != text

    with pytest.raises(AssertionError):
        _check_caller(CALLERS[path], removed, hits)


def test_the_helper_layering_follows_the_export_call_inside_resolve_compose_file_args() -> (
    None
):
    library = COMPOSE_FILES_LIB.read_text(encoding="utf-8")
    body = re.search(
        r"^resolve_compose_file_args\(\) \{\n(.*?)^\}", library, re.M | re.S
    )
    assert body is not None
    assert "export_lane_overlay_dir" in body.group(1)
    assert body.group(1).index("export_lane_overlay_dir") < body.group(1).index(
        "docker-compose.dev-lane.yml"
    )
    for helper in ("resolve_lane_overlay_dir", "export_lane_overlay_dir"):
        helper_body = re.search(rf"^{helper}\(\) \{{\n(.*?)^\}}", library, re.M | re.S)
        assert helper_body is not None, helper
        assert "_pin_lane_declaration_dir" in helper_body.group(1), helper


def test_the_workflows_that_name_a_dev_lane_file_run_no_compose_command_on_it() -> None:
    # The two workflow readers parse the file as YAML; the deriver imports no subprocess, so
    # no hosted job renders these files and none needs the variable.
    for path in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            named = any(name in line for name in LAYERING_FILES)
            runs_compose = re.search(r"docker\s+compose\b|\bdocker-compose\s", line)
            assert not (named and runs_compose), f"{path.name}:{number}"
    deriver = REPO_ROOT / "scripts" / "runtime_build" / "declared_consumer_groups.py"
    assert "subprocess" not in deriver.read_text(encoding="utf-8")


def test_no_workflow_clears_the_ci_variables_the_compose_fixture_reads() -> None:
    # The `compose_cli` fixture treats CI and GITHUB_ACTIONS as "this is a hosted run".
    clearing = re.compile(
        r"^\s*(?:CI|GITHUB_ACTIONS)\s*:\s*(?:\"\"|''|false|\"false\"|0)?\s*$"
        r"|unset\s+(?:CI|GITHUB_ACTIONS)\b|env\s+-u\s+(?:CI|GITHUB_ACTIONS)\b",
        re.M,
    )
    offenders = sorted(
        path.name
        for path in (REPO_ROOT / ".github").rglob("*.y*ml")
        if clearing.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == []


def _command_after_env_prefix(value: str) -> list[str]:
    tokens = shlex.split(value)
    assert tokens[0] == "env"
    i = 1
    while i < len(tokens) and "=" in tokens[i] and tokens[i].split("=", 1)[0].isupper():
        i += 1
    return tokens[i:]


def test_the_read_command_is_one_the_ledger_cli_accepts() -> None:
    tokens = _command_after_env_prefix(LEDGER_ENV["ONEX_LEDGER_BUS_READ_COMMAND"])
    assert tokens[:3] == ["python", "-m", "omnibase_core.cli.cli_commands"]
    assert tokens[3:] == ["ledger", "query", "--bus-lane", "dev", "--stdin"]
    ctx = ledger_query.make_context("query", tokens[5:])
    assert ctx.params["read_stdin"] is True
    assert ctx.params["bus_lane"] == "dev"


def test_the_append_command_names_dev_and_the_registry_root_the_read_command_names() -> (
    None
):
    tokens = _command_after_env_prefix(LEDGER_ENV["ONEX_LEDGER_BUS_APPEND_COMMAND"])
    assert tokens == [
        "python",
        "-m",
        "omnimarket.work_ledger_bus.cli",
        "append",
        "--bus-lane",
        "dev",
        "--omnibase-path",
        "/app",
    ]
    read_tokens = shlex.split(LEDGER_ENV["ONEX_LEDGER_BUS_READ_COMMAND"])
    read_prefix = [t for t in read_tokens if t.startswith("OMNIBASE_PATH=")]
    assert read_prefix == ["OMNIBASE_PATH=/app"]
    assert tokens[-1] == "/app"


def test_control_a_renamed_read_option_is_refused_by_the_cli() -> None:
    with pytest.raises(click.NoSuchOption):
        ledger_query.make_context("query", ["--bus-lane", "dev", "--from-stdin"])


def test_control_a_read_command_for_another_lane_is_refused_by_the_cli() -> None:
    with pytest.raises(click.BadParameter):
        ledger_query.make_context("query", ["--bus-lane", "stability", "--stdin"])
