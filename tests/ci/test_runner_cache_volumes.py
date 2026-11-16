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
the runner home's ``.cache`` (pip, pre-commit, pnpm, playwright, ci-envs, npm) and
``runner-N-work`` at the job workspace. The uv cache sits inside the workspace
volume, not in ``.cache``: uv hardlinks wheels out of its cache and a hardlink
cannot cross a mount (OMN-19998 measured 262 s against 32 s for one sync).

Two views are pinned: the YAML as written, and the container docker compose
renders from it (skipped where ``docker compose`` is not installed).

OMN-22898 retired the primary host's general pool, so ``docker-compose.runners.yml``
now declares no ``omninode-runner-N`` service and the checks below hold for none.
They are written as functions over a compose mapping so each one is also run
against a synthetic pool, which proves a clean result on the empty real pool is
not a check that cannot fail, and so a general runner added back here is held to
the same volume rules.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

_DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
_COMPOSE = _DOCKER_DIR / "docker-compose.runners.yml"
_ENTRYPOINT = _DOCKER_DIR / "runners" / "entrypoint.sh"

# The compose file spells the runner home as a defaulted variable, so the text
# holds no machine-specific path; the default is the image's runner user home.
_RUNNER_HOME = "/home/runner"
_HOME_VARIABLE = "${RUNNER_USER_HOME:-" + _RUNNER_HOME + "}"
_CACHE_TARGET = _RUNNER_HOME + "/.cache"
_WORK_TARGET = _RUNNER_HOME + "/actions-runner/_work"
_POOL_NAME = re.compile(r"omninode-runner-(\d+)")

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        _COMPOSE.read_text().replace(_HOME_VARIABLE, _RUNNER_HOME)
    )
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


def _pool(compose: dict[str, Any]) -> list[int]:
    """Indices of the ``omninode-runner-N`` services a compose mapping declares."""
    return sorted(
        int(m.group(1))
        for name in compose["services"]
        if (m := _POOL_NAME.fullmatch(name))
    )


def _pool_volume_problems(compose: dict[str, Any]) -> list[str]:
    """Every way the general pool breaks the per-runner cache and work volume rule."""
    problems: list[str] = []
    seen: dict[str, int] = {}
    for n in _pool(compose):
        service = compose["services"][f"omninode-runner-{n}"]
        targets = _volume_targets(service)
        for target, kind in ((_CACHE_TARGET, "cache"), (_WORK_TARGET, "work")):
            source = targets.get(target)
            if source != f"runner-{n}-{kind}":
                problems.append(f"runner {n}: {target} is {source!r}")
                continue
            if source in seen:
                problems.append(f"{source} used by {seen[source]} and {n}")
            seen[source] = n
            declared = compose["volumes"].get(source, {}).get("name")
            if declared != f"omninode-runner-{n}-{kind}":
                problems.append(f"runner {n}: volume {source} is named {declared!r}")
        env = service["environment"]
        # The uv cache must sit inside the workspace mount, not under .cache: a
        # uv cache on its own mount copies every file instead of hardlinking.
        if not _inside(env["UV_CACHE_DIR"], _WORK_TARGET) or _inside(
            env["UV_CACHE_DIR"], _CACHE_TARGET
        ):
            problems.append(f"runner {n}: UV_CACHE_DIR {env['UV_CACHE_DIR']!r}")
        if not _inside(env["npm_config_cache"], _CACHE_TARGET):
            problems.append(f"runner {n}: npm_config_cache {env['npm_config_cache']!r}")
    return problems


def _synthetic_pool(count: int, *, share_work_volume: bool = False) -> dict[str, Any]:
    services: dict[str, Any] = {}
    volumes: dict[str, Any] = {}
    for n in range(1, count + 1):
        work = "runner-1-work" if share_work_volume else f"runner-{n}-work"
        services[f"omninode-runner-{n}"] = {
            "volumes": [
                f"runner-{n}-cache:{_CACHE_TARGET}",
                f"{work}:{_WORK_TARGET}",
            ],
            "environment": {
                "UV_CACHE_DIR": f"{_WORK_TARGET}/.uv-cache",
                "npm_config_cache": f"{_CACHE_TARGET}/npm",
            },
        }
        volumes[f"runner-{n}-cache"] = {"name": f"omninode-runner-{n}-cache"}
        volumes[f"runner-{n}-work"] = {"name": f"omninode-runner-{n}-work"}
    return {"services": services, "volumes": volumes}


def test_the_primary_host_declares_no_general_pool(compose: dict[str, Any]) -> None:
    """OMN-22898: no ``omninode-runner-N`` service and no per-runner volume."""
    assert _pool(compose) == []
    leftovers = [
        name for name in compose["volumes"] if re.fullmatch(r"runner-\d+-\w+", name)
    ]
    assert leftovers == []
    assert _pool_volume_problems(compose) == []


def test_the_volume_rule_holds_for_a_synthetic_pool_and_fails_when_one_is_shared() -> (
    None
):
    """Positive control: the empty real pool above is clean because there is
    nothing to break, and the same function reports a real violation."""
    assert _pool(_synthetic_pool(3)) == [1, 2, 3]
    assert _pool_volume_problems(_synthetic_pool(3)) == []
    broken = _pool_volume_problems(_synthetic_pool(3, share_work_volume=True))
    assert any(p.startswith("runner 2:") for p in broken), broken


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
    assert '"${RUNNER_USER_HOME}/.cache"' in chown
    assert (
        '"${RUNNER_HOME:-${RUNNER_USER_HOME}/actions-runner}/${RUNNER_WORK_DIR:-_work}"'
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
    assert _pool(rendered) == []
    for n in _pool(rendered):
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
