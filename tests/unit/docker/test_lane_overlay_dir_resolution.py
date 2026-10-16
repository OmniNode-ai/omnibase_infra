# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20932: lane overlays are private config, mounted from ONEX_LANE_OVERLAY_DIR.

Operator RULING 2026-10-10T21:57:38Z: no lab or deployment configuration lives
in a shipped repository. The lab lane overlays left ``docker/lane-overlays/``;
this repository keeps the typed schema, the renderer and the neutral
``local.bifrost.example.yaml`` template. Every lane compose file mounts its
overlay from ``${ONEX_LANE_OVERLAY_DIR:?...}``, so an unset directory is a loud
compose refusal naming how to supply one, never a silent mount of a lab value.
``compose_files.sh`` resolves the directory for every shell deploy path.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.lane_overlay_fixtures import (
    LANE_OVERLAY_DIR_MESSAGE,
    LANE_OVERLAY_MOUNT,
    LANE_OVERLAY_TARGET,
    lane_overlay_mounts,
    load_compose,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
COMPOSE_FILES_SH = ROOT / "scripts" / "runtime_build" / "compose_files.sh"
_JUDGE_COMPOSE = DOCKER / "docker-compose.judge.yml"
_COMPOSE_PROJECT = f"omnibase-infra-{_JUDGE_COMPOSE.stem.partition('.')[2]}"
_RUNTIME_SERVICES = {"omninode-runtime", "runtime-effects"}
_WORKER_SERVICES = {"runtime-worker", "tenant-projection-writer"}


def _lane_mount_services(lane: str) -> set[str]:
    """The services a lane recipe lists the overlay bind on.

    Layered worker lanes carry four runtimes. The prod recipe restates the
    volumes of main, effects and worker only; its tenant projection writer
    inherits the base file's bind (OMN-17101).
    """
    if lane in {"infra", "prepr"} or lane.startswith(("dev-", "stability")):
        return _RUNTIME_SERVICES | _WORKER_SERVICES
    if lane == "prod":
        return _RUNTIME_SERVICES | {"runtime-worker"}
    return set(_RUNTIME_SERVICES)


#: Recipes declaring the shared bind, and the services each lists it on.
_LANE_COMPOSE_MOUNTS = {
    path.name: _lane_mount_services(path.stem.partition(".")[2])
    for path in sorted(DOCKER.glob("docker-compose*.yml"))
    if "x-lane-overlay-mount" in load_compose(path)
}


def _values_for_key(data: Any, key: str) -> Iterator[Any]:
    if isinstance(data, dict):
        for name, value in data.items():
            if name == key:
                yield value
            yield from _values_for_key(value, key)
    elif isinstance(data, list):
        for value in data:
            yield from _values_for_key(value, key)


def test_the_shipped_overlay_directory_holds_only_the_neutral_template() -> None:
    shipped = sorted(path.name for path in (DOCKER / "lane-overlays").iterdir())
    lab = [
        name
        for name in shipped
        if name.endswith(".bifrost.yaml") or name.startswith("dev.rsd-")
    ]
    assert lab == []
    assert "local.bifrost.example.yaml" in shipped


def test_all_lane_compose_recipes_are_covered() -> None:
    assert sorted(len(services) for services in _LANE_COMPOSE_MOUNTS.values()) == (
        [2] * 4 + [3] + [4] * 6
    )
    assert "docker-compose.prod.yml" in _LANE_COMPOSE_MOUNTS


@pytest.mark.parametrize(
    ("compose_name", "services"),
    sorted(_LANE_COMPOSE_MOUNTS.items()),
    ids=sorted(_LANE_COMPOSE_MOUNTS),
)
def test_every_lane_compose_file_mounts_its_overlay_from_the_private_directory(
    compose_name: str, services: set[str]
) -> None:
    path = DOCKER / compose_name
    assert lane_overlay_mounts(path) == {
        service: [LANE_OVERLAY_MOUNT] for service in services
    }
    values = list(_values_for_key(load_compose(path), "BIFROST_LANE_OVERLAY_PATH"))
    assert all(value in (LANE_OVERLAY_TARGET, "") for value in values)


def test_no_compose_file_mounts_a_committed_lane_overlay() -> None:
    offenders = {
        compose.name: sources
        for compose in sorted(DOCKER.glob("docker-compose*.yml"))
        if (
            sources := [
                volume.get("source", "") if isinstance(volume, dict) else volume
                for volumes in _values_for_key(load_compose(compose), "volumes")
                if isinstance(volumes, list)
                for volume in volumes
                if "./lane-overlays/" in str(volume)
            ]
        )
    }
    assert offenders == {}


def test_the_mount_message_names_both_variables() -> None:
    assert "ONEX_LANE_OVERLAY_DIR" in LANE_OVERLAY_DIR_MESSAGE
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in LANE_OVERLAY_DIR_MESSAGE
    assert "}" not in LANE_OVERLAY_DIR_MESSAGE


def _resolve(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = (
        f"source {COMPOSE_FILES_SH}; "
        "declare -a args; "
        f'resolve_compose_file_args args "{ROOT}" "{_COMPOSE_PROJECT}" || exit $?; '
        'printf "%s\\n" "${ONEX_LANE_OVERLAY_DIR}"; printf "%s\\n" "${args[@]}"'
    )
    base = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"ONEX_LANE_OVERLAY_DIR", "ONEX_WORKSPACE_CONFIG_ROOT", "OMNI_HOME"}
    }
    return subprocess.run(
        ["bash", "-c", script],
        env={**base, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_compose_helper_resolves_the_config_root_directory(tmp_path: Path) -> None:
    overlays = tmp_path / "owner" / "config" / "lab" / "lane-overlays"
    overlays.mkdir(parents=True)
    (overlays / f"{_COMPOSE_PROJECT}.bifrost.yaml").write_text("{}")
    result = _resolve({"ONEX_WORKSPACE_CONFIG_ROOT": str(tmp_path / "owner")})
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == str(overlays.resolve())
    assert lines[1:] == [
        "-f",
        str(DOCKER / "docker-compose.infra.yml"),
        "-f",
        str(DOCKER / "docker-compose.judge.yml"),
    ]


def test_compose_helper_keeps_an_explicit_directory(tmp_path: Path) -> None:
    explicit = tmp_path / "mine"
    explicit.mkdir()
    (explicit / f"{_COMPOSE_PROJECT}.bifrost.yaml").write_text("{}")
    (tmp_path / "owner" / "config" / "lab" / "lane-overlays").mkdir(parents=True)
    result = _resolve(
        {
            "ONEX_LANE_OVERLAY_DIR": str(explicit),
            "ONEX_WORKSPACE_CONFIG_ROOT": str(tmp_path / "owner"),
        }
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == str(explicit.resolve())


def test_compose_helper_unsets_a_directory_without_the_project_overlay(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "mine"
    explicit.mkdir()
    (explicit / "other-project.bifrost.yaml").write_text("{}")
    result = _resolve({"ONEX_LANE_OVERLAY_DIR": str(explicit)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "",
        "-f",
        str(DOCKER / "docker-compose.infra.yml"),
        "-f",
        str(_JUDGE_COMPOSE),
    ]


_NO_DIRECTORY_ENVS = [
    {},
    {"OMNI_HOME": "/nonexistent/workspace"},
    {"ONEX_WORKSPACE_CONFIG_ROOT": "/nonexistent/owner"},
    {"ONEX_LANE_OVERLAY_DIR": "/nonexistent/overlays"},
]
_NO_DIRECTORY_IDS = [
    "nothing-set",
    "no-sibling",
    "no-config-root-dir",
    "missing-explicit-dir",
]


@pytest.mark.parametrize("env", _NO_DIRECTORY_ENVS, ids=_NO_DIRECTORY_IDS)
def test_compose_helper_leaves_the_directory_unset_for_compose_to_refuse(
    env: dict[str, str],
) -> None:
    """With no directory, the mount's own ``:?`` message is the refusal."""
    result = _resolve(env)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == ""
    assert lines[1:] == [
        "-f",
        str(DOCKER / "docker-compose.infra.yml"),
        "-f",
        str(DOCKER / "docker-compose.judge.yml"),
    ]


def _resolve_strict(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = (
        f"source {COMPOSE_FILES_SH}; "
        f'resolve_lane_overlay_dir "{_COMPOSE_PROJECT}" || exit $?; '
        'printf "%s\\n" "${ONEX_LANE_OVERLAY_DIR}"'
    )
    base = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"ONEX_LANE_OVERLAY_DIR", "ONEX_WORKSPACE_CONFIG_ROOT", "OMNI_HOME"}
    }
    return subprocess.run(
        ["bash", "-c", script],
        env={**base, **env},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("env", _NO_DIRECTORY_ENVS, ids=_NO_DIRECTORY_IDS)
def test_strict_helper_refuses_without_an_overlay_directory(
    env: dict[str, str],
) -> None:
    result = _resolve_strict(env)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "ONEX_LANE_OVERLAY_DIR" in result.stderr
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in result.stderr


def test_strict_helper_resolves_the_config_root_directory(tmp_path: Path) -> None:
    overlays = tmp_path / "owner" / "config" / "lab" / "lane-overlays"
    overlays.mkdir(parents=True)
    (overlays / f"{_COMPOSE_PROJECT}.bifrost.yaml").write_text("{}")
    result = _resolve_strict({"ONEX_WORKSPACE_CONFIG_ROOT": str(tmp_path / "owner")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(overlays.resolve())]


def test_strict_helper_refuses_a_directory_without_the_project_overlay(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "mine"
    explicit.mkdir()
    (explicit / "other-project.bifrost.yaml").write_text("{}")
    result = _resolve_strict({"ONEX_LANE_OVERLAY_DIR": str(explicit)})
    assert result.returncode == 1
    assert result.stdout == ""
    assert "ONEX_LANE_OVERLAY_DIR" in result.stderr
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in result.stderr
