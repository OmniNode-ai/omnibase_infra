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

import posixpath

import pytest

from tests.ci._compose_render_omn21099 import (
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
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

needs_compose = pytest.mark.skipif(
    not docker_compose_available(),
    reason="docker compose CLI absent; the committed-entry checks still hold",
)


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


@needs_compose
def test_runtime_effects_renders_the_mount_source() -> None:
    services = render_stack(
        DEV_LANE_STACK,
        {"ONEX_LANE_OVERLAY_DIR": "/srv/workspace/config/lab/lane-overlays"},
    )
    volumes = services["runtime-effects"]["volumes"]
    mounts = [v for v in volumes if v["target"] == LEDGER_MOUNT_TARGET]
    # Compose keeps the dots; the engine's mount resolves them, and the lane
    # declaration folder (the parent of the lane overlay folder) is where they land.
    assert [posixpath.normpath(m["source"]) for m in mounts] == [
        "/srv/workspace/config/lab"
    ]
    assert mounts[0]["read_only"] is True
