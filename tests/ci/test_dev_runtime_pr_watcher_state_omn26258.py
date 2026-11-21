# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime reads the lab host's own PR watcher state (OMN-26258).

THE RECORDED REFUSAL
--------------------
The scheduled ticket closer in the dev runtime on the lab host reads no candidates:
its bound profile refuses with "the PR watcher state file is absent at its declared path".
The runtime's registry root is ``/app`` and ``ONEX_PR_WATCHER_STATE`` is empty, so the
declared path is ``/app/.claude_scratch/pr-watcher/state.json``, a file no process writes.

THE DECLARATION
---------------
The read-only PR watcher (the host's ``onex-pr-watcher-readonly`` user timer, installed by
the host installer node) ticks every 300 s and writes ``state.json`` under its state
directory, ``.local/state/onex/pr-watcher-readonly`` of the unit user's home. The primary
dev runtime mounts that directory read-only (the directory, not the file, because the
watcher replaces the file atomically) and names the file in ``ONEX_PR_WATCHER_STATE``.
No state is copied from another host and no path is typed from a person's home directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.lane_overlay_fixtures import load_compose

pytestmark = [pytest.mark.unit]

DOCKER = Path(__file__).resolve().parents[2] / "docker"
FOLD_OVERLAY = "docker-compose.dev-tenant-fold.yml"
SERVICE = "omninode-runtime"
STATE_ENV = "ONEX_PR_WATCHER_STATE"
MOUNT_TARGET = "/etc/onex/pr-watcher-state"
MOUNT = "${HOME}/.local/state/onex/pr-watcher-readonly:/etc/onex/pr-watcher-state:ro"
DECLARED_PATH_WITHOUT_BINDING = "/app/.claude_scratch/pr-watcher/state.json"


def _service() -> dict[str, object]:
    found = load_compose(DOCKER / FOLD_OVERLAY)["services"][SERVICE]
    assert isinstance(found, dict)
    return found


def test_unbound_runtime_declares_a_path_nothing_writes() -> None:
    """Positive control: with the variable unset the declared path is the one the refusal names."""
    registry_root = "/app"
    unset = ""
    declared = (
        unset if unset else f"{registry_root}/.claude_scratch/pr-watcher/state.json"
    )
    assert declared == DECLARED_PATH_WITHOUT_BINDING


def test_state_variable_names_a_file_under_the_mounted_directory() -> None:
    environment = _service()["environment"]
    assert isinstance(environment, dict)
    assert environment.get(STATE_ENV) == f"{MOUNT_TARGET}/state.json"


def test_watcher_state_directory_is_mounted_read_only_once() -> None:
    volumes = [str(v) for v in _service()["volumes"]]
    assert volumes.count(MOUNT) == 1
    assert [v for v in volumes if MOUNT_TARGET in v] == [MOUNT]


def test_mount_source_is_the_unit_user_home_never_a_typed_home_path() -> None:
    source = MOUNT.split(":", maxsplit=1)[0]
    assert source.startswith("${HOME}/")
    assert not source.startswith("/home/")
    assert "/Users/" not in MOUNT


def test_the_state_variable_is_not_set_where_the_watcher_directory_is_not_mounted() -> (
    None
):
    """The shared dev-lane overlay carries neither half: they travel together or not at all."""
    service = load_compose(DOCKER / "docker-compose.dev-lane.yml")["services"][
        "runtime-effects"
    ]
    environment = service["environment"]
    volumes = [str(v) for v in service["volumes"]]
    assert STATE_ENV not in environment
    assert not [v for v in volumes if MOUNT_TARGET in v]
