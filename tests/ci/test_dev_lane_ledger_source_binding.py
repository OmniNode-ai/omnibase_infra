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
from typing import Any

import click
import pytest

from omnibase_infra.cli.cli_ledger import ledger_query
from tests.ci._compose_render_omn21099 import (
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    DOCKER_DIR,
    docker_compose_available,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

CONFIG_ROOT_PREFIX = "env ONEX_WORKSPACE_CONFIG_ROOT=/etc/onex/workspace-config"
LEDGER_ENV = {
    "ONEX_LEDGER_BUS_READ_COMMAND": (
        f"{CONFIG_ROOT_PREFIX} OMNIBASE_PATH=/app onex ledger query --bus-lane dev --stdin"
    ),
    "ONEX_LEDGER_BUS_APPEND_COMMAND": (
        f"{CONFIG_ROOT_PREFIX} python -m omnimarket.work_ledger_bus.cli append "
        "--bus-lane dev --omnibase-path /app"
    ),
    "ONEX_LEDGER_HOST_NAME": "${DEV_LANE_LEDGER_HOST_NAME:-dev-runtime}",
}
LEDGER_MOUNT_TARGET = "/etc/onex/workspace-config/config/lab"

REQUIRED_SOURCE_PREFIX = "${ONEX_LANE_DECLARATION_DIR:?"


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
    require_compose(docker_compose_available(), bool(os.environ.get("CI")))


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
        f"{path.name}:{number}"
        for path in DOCKER_DIR.glob("*.yml")
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


def _command_after_env_prefix(value: str) -> list[str]:
    tokens = shlex.split(value)
    assert tokens[0] == "env"
    i = 1
    while i < len(tokens) and "=" in tokens[i] and tokens[i].split("=", 1)[0].isupper():
        i += 1
    return tokens[i:]


def test_the_read_command_is_one_the_ledger_cli_accepts() -> None:
    tokens = _command_after_env_prefix(LEDGER_ENV["ONEX_LEDGER_BUS_READ_COMMAND"])
    assert tokens == ["onex", "ledger", "query", "--bus-lane", "dev", "--stdin"]
    ctx = ledger_query.make_context("query", tokens[3:])
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
