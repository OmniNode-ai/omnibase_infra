# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21807: the general-pool runners keep their tool caches on named volumes.

Incident (lane dev-restore-1515, TERMINAL 2026-10-31T16:03:34Z). Each runner
recreate removed a container whose writable layer held the uv cache and the shared
ci-envs. containerd's garbage collector held its metadata write lock for the whole
delete, about 28 minutes, and every ``docker rm`` on the host blocked behind it, so
the dev deploy agent's redeploys failed.

The fix is that a recreate removes a thin layer. Every ``omninode-runner-N``
service mounts two per-runner named volumes: ``runner-N-cache`` at
``/home/runner/.cache`` (pip, pre-commit, pnpm, playwright, ci-envs, npm) and
``runner-N-work`` at the job workspace. The uv cache sits inside the workspace
volume, not in ``.cache``: uv hardlinks wheels out of its cache and a hardlink
cannot cross a mount (OMN-19998 measured 262 s against 32 s for one sync).

Two views are pinned: the YAML as written, and the container docker compose
renders from it (skipped where ``docker compose`` is not installed).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

_DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
_COMPOSE = _DOCKER_DIR / "docker-compose.runners.yml"
_ENTRYPOINT = _DOCKER_DIR / "runners" / "entrypoint.sh"

_CACHE_TARGET = "/home/runner/.cache"
_WORK_TARGET = "/home/runner/actions-runner/_work"
_POOL = range(1, 11)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(_COMPOSE.read_text())
    return loaded


def _volume_targets(service: dict[str, Any]) -> dict[str, str]:
    """target -> source for a service's short-syntax volume entries."""
    out: dict[str, str] = {}
    for entry in service["volumes"]:
        if isinstance(entry, str):
            parts = entry.split(":")
            out[parts[1]] = parts[0]
    return out


def _inside(path: str, root: str) -> bool:
    return PurePosixPath(root) in PurePosixPath(path).parents


@pytest.mark.parametrize("n", _POOL)
def test_each_pool_runner_mounts_its_own_cache_and_work_volumes(
    compose: dict[str, Any], n: int
) -> None:
    targets = _volume_targets(compose["services"][f"omninode-runner-{n}"])
    assert targets[_CACHE_TARGET] == f"runner-{n}-cache"
    assert targets[_WORK_TARGET] == f"runner-{n}-work"


@pytest.mark.parametrize("n", _POOL)
def test_the_cache_and_work_volumes_are_declared_with_stable_names(
    compose: dict[str, Any], n: int
) -> None:
    volumes = compose["volumes"]
    assert volumes[f"runner-{n}-cache"]["name"] == f"omninode-runner-{n}-cache"
    assert volumes[f"runner-{n}-work"]["name"] == f"omninode-runner-{n}-work"


def test_no_volume_is_shared_between_runners(compose: dict[str, Any]) -> None:
    """A shared cache volume would let one runner's job mutate another's."""
    seen: dict[str, str] = {}
    for n in _POOL:
        targets = _volume_targets(compose["services"][f"omninode-runner-{n}"])
        for target in (_CACHE_TARGET, _WORK_TARGET):
            source = targets[target]
            assert source not in seen, f"{source} used by {seen[source]} and {n}"
            seen[source] = f"omninode-runner-{n}"


def test_the_pool_is_ten_runners(compose: dict[str, Any]) -> None:
    pool = [name for name in compose["services"] if name.startswith("omninode-runner-")]
    assert sorted(pool) == sorted(f"omninode-runner-{n}" for n in _POOL)


def test_uv_cache_lives_inside_the_workspace_volume_so_uv_still_hardlinks(
    compose: dict[str, Any],
) -> None:
    """Negative control for the hardlink hazard: a uv cache on its own mount copies
    every file. UV_CACHE_DIR must sit inside the workspace mount, not under .cache."""
    for n in _POOL:
        service = compose["services"][f"omninode-runner-{n}"]
        uv_cache = service["environment"]["UV_CACHE_DIR"]
        assert _inside(uv_cache, _WORK_TARGET), uv_cache
        assert not _inside(uv_cache, _CACHE_TARGET), uv_cache
        # and the workspace is mounted, so the cache is on that volume
        assert _WORK_TARGET in _volume_targets(service)


def test_npm_cache_is_under_the_cache_volume(compose: dict[str, Any]) -> None:
    for n in _POOL:
        env = compose["services"][f"omninode-runner-{n}"]["environment"]
        assert _inside(env["npm_config_cache"], _CACHE_TARGET)


def test_the_entrypoint_hands_a_root_owned_mount_to_the_runner_user() -> None:
    """The image has no workspace directory, so Docker creates the mount point
    root-owned on first use; the root phase of the entrypoint must hand it over
    before it drops to the runner user."""
    text = _ENTRYPOINT.read_text()
    root_phase = text[
        text.index('if [[ "$(id -u)" -eq 0 ]]; then\n    _fix_docker_socket_gid') :
    ]
    root_phase = root_phase[: root_phase.index("\nfi\n")]
    chown = next(line for line in root_phase.splitlines() if "chown" in line)
    assert "chown runner:runner" in chown
    assert "/home/runner/.cache" in chown
    assert (
        '"${RUNNER_HOME:-/home/runner/actions-runner}/${RUNNER_WORK_DIR:-_work}"'
        in chown
    )
    assert " -R" not in chown, (
        "a recursive chown of a cache volume costs minutes per start"
    )
    # the root phase comes before the runner is configured or run
    assert text.index(chown) < text.index('_as_runner "${RUNNER_HOME}/config.sh"')
    assert text.index(chown) < text.index('_as_runner "${RUNNER_HOME}/run.sh"')


_RENDER_ENV = {
    "DEPLOY_RUNNER_OMNI_HOME": "/placeholder",
    "DEPLOY_RUNNER_OPERATOR_ENV_FILE": "/placeholder",
    "LLM_ENDPOINT_CIDR_ALLOWLIST": "cidr-placeholder",
    "LOCAL_LLM_SHARED_SECRET": "placeholder-not-a-secret",
    "OMNI_HOME": "/placeholder",
    "DEPLOY_RUNNER_TOKEN": "placeholder-not-a-token",
    "RUNNER_TOKEN": "placeholder-not-a-token",
}


def _render() -> dict[str, Any] | None:
    if shutil.which("docker") is None:
        return None
    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", str(_COMPOSE), "config", "--format", "json"],
            env={**os.environ, **_RENDER_ENV},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    rendered: dict[str, Any] = json.loads(proc.stdout)
    return rendered


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_a_rendered_runner_container_has_its_caches_on_named_volumes() -> None:
    rendered = _render()
    if rendered is None:
        pytest.skip("docker compose config could not render the file here")
    for n in _POOL:
        service = rendered["services"][f"omninode-runner-{n}"]
        mounts = {v["target"]: v for v in service["volumes"]}
        for target, volume in (
            (_CACHE_TARGET, f"runner-{n}-cache"),
            (_WORK_TARGET, f"runner-{n}-work"),
        ):
            assert mounts[target]["type"] == "volume", target
            assert mounts[target]["source"] == volume
        env = service["environment"]
        assert _inside(env["UV_CACHE_DIR"], _WORK_TARGET)
        assert _inside(env["npm_config_cache"], _CACHE_TARGET)
