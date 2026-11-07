# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's runtime-effects carries the ticket-creation guard (OMN-22691).

``node_linear_ticket_write_effect`` runs in the dev runtime-effects container and judges every
create with the omniclaude ticket-creation guard. ``node_create_ticket`` locates it through
``ONEX_TICKET_GUARD_PATH`` and refuses a create without it, so a container with neither the
variable nor the file answered every lab lane's ``onex ticket create`` with ``guard_unavailable``.
The guard loads its lane-intake handler by path from the clone root, so the mount is the whole
omniclaude clone, read-only, taken from the clone the skills mount already comes from.

Those readers live in omnimarket and omniclaude, which this repository does not install, so these
checks cover the shipped compose configuration: the committed entries and the value compose renders
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

GUARD_ENV = "ONEX_TICKET_GUARD_PATH"
GUARD_ROOT_IN_CONTAINER = "/etc/onex/ticket-guard"
GUARD_FILE = "plugins/onex/hooks/lib/ticket_creation_guard.py"
SKILLS_VARIABLE = "OMNICLAUDE_SKILLS_DIR"

needs_compose = pytest.mark.skipif(
    not docker_compose_available(),
    reason="docker compose CLI absent; the committed-entry checks still hold",
)


def _services() -> dict[str, dict[str, object]]:
    services = load_compose(DEV_LANE_OVERLAY)["services"]
    assert isinstance(services, dict)
    return services


def _mounts_of(service: dict[str, object]) -> list[str]:
    volumes = service.get("volumes", [])
    assert isinstance(volumes, list)
    return [str(v) for v in volumes if GUARD_ROOT_IN_CONTAINER in str(v)]


def _environment_of(service: dict[str, object]) -> dict[str, object]:
    env = service.get("environment", {})
    return env if isinstance(env, dict) else {}


def test_runtime_effects_names_the_guard_file_inside_the_mounted_clone() -> None:
    env = _environment_of(_services()["runtime-effects"])
    assert env[GUARD_ENV] == f"{GUARD_ROOT_IN_CONTAINER}/{GUARD_FILE}"


def test_runtime_effects_mounts_the_clone_root_read_only_from_the_skills_directory() -> (
    None
):
    mounts = _mounts_of(_services()["runtime-effects"])
    assert mounts == [
        f"${{{SKILLS_VARIABLE}:-./skills}}/../../..:{GUARD_ROOT_IN_CONTAINER}:ro"
    ]


def test_no_other_dev_lane_service_gains_the_mount() -> None:
    holders = sorted(
        name
        for name, service in _services().items()
        if isinstance(service, dict)
        and (_mounts_of(service) or GUARD_ENV in _environment_of(service))
    )
    assert holders == ["runtime-effects"]


@needs_compose
def test_runtime_effects_renders_the_clone_root_as_the_mount_source() -> None:
    services = render_stack(
        DEV_LANE_STACK,
        {SKILLS_VARIABLE: "/data/omninode/omniclaude/plugins/onex/skills"},
    )
    volumes = services["runtime-effects"]["volumes"]
    mounts = [v for v in volumes if v["target"] == GUARD_ROOT_IN_CONTAINER]
    # Compose keeps the dots; the engine's mount resolves them, and the clone root is where they land.
    assert [posixpath.normpath(m["source"]) for m in mounts] == [
        "/data/omninode/omniclaude"
    ]
    assert mounts[0]["read_only"] is True
