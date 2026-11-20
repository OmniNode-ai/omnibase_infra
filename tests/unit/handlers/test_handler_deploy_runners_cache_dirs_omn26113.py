# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26113: the prepare effect creates every cache bind source of a pool runner-owned, declared job root or not.

The h051 pools declare no ``job_dir_host_root`` (and a pool's prefix never equals its host row's), so the effect of
OMN-26093 left ``cache/npm``, ``cache/pip`` and the rest to Docker, which creates a missing bind source root-owned
while every job runs as uid 1001: the first npm or pre-commit call died with EACCES. The effect now reads the cache
sources from the rendered Compose (a bind whose target is ``~/.npm`` or under ``~/.cache``) and prepares them as the
runner user with a readback, as it does the TMPDIR.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_deploy_runners import HandlerDeployRunners
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FLEET = ROOT / "config" / "runner_fleet.yaml"
HOME = "/home/runner"
SCRATCH = "tmp"
CACHE_NAMES = ("pip", "pre-commit", "pnpm", "ms-playwright", "ci-envs", "npm")


@dataclass(frozen=True)
class Pool:
    compose: str
    host: str
    prefix: str
    count: int
    cache_base: str  # `<base>/<N>/<name>` or `<base>/<N>/cache/<name>`


OMNIBASE_POOL = Pool(
    "docker-compose.runners-hp051-omnibase-ci-runner.yml",
    "h051",
    "hp051-omnibase-ci-runner",
    16,
    "/mnt/h051-nvme/runners/hp051-omnibase-ci-runner/{n}/cache/{name}",
)
SCRATCH_POOL = Pool(
    "docker-compose.runners-hp051-ci-runner.yml",
    "h051",
    "hp051-ci-runner",
    4,
    "/data/scratch/cache/runners/{n}/{name}",
)


def _rendered_compose(pool: Pool) -> str:
    """What ``docker compose config --format json`` prints for the shipped pool: long-form binds, home expanded."""
    document = yaml.safe_load((ROOT / "docker" / pool.compose).read_text("utf-8"))
    services: dict[str, Any] = {}
    for name, service in document["services"].items():
        volumes = []
        for volume in service["volumes"]:
            if isinstance(volume, str) and volume.startswith("/"):
                source, _, target = volume.replace(
                    "${RUNNER_USER_HOME:-/home/runner}", HOME
                ).partition(":")
                volumes.append(
                    {
                        "type": "bind",
                        "source": source,
                        "target": target.split(":")[0],
                    }
                )
        services[name] = {
            "environment": {"TMPDIR": f"{HOME}/{SCRATCH}"},
            "volumes": volumes,
        }
    return json.dumps({"services": services})


class FakeHostHandler(HandlerDeployRunners):
    """ssh replaced by an in-memory host: ``install -d`` records owner and mode, ``stat`` reads them back."""

    def __init__(self, pool: Pool, *, root_owned: frozenset[str] = frozenset()) -> None:
        self.dirs: dict[str, tuple[int, str]] = {}
        self._rendered = _rendered_compose(pool)
        self._root_owned = root_owned

    async def _converge_lab_host_table(
        self, request: ModelDeployRunnersRequest, timeout: int
    ) -> str | None:
        return None  # OMN-26098: covered by its own test

    async def _remote(self, host: str, command: str, timeout: int) -> str:
        if "compose" in command and "config" in command:
            return self._rendered
        if " install " in f" {command} ":
            tokens = shlex.split(command.split("; ")[-1])
            path = tokens[-1]
            owner = (
                0 if path in self._root_owned else int(tokens[tokens.index("-o") + 1])
            )
            self.dirs[path] = (owner, tokens[tokens.index("-m") + 1])
            return ""
        match = re.search(r"stat -c '%F:%u:%a' -- (\S+)", command)
        assert match is not None, command
        owner, mode = self.dirs[shlex.split(match.group(1))[0]]
        return f"directory:{owner}:{mode}\n"


def _request(pool: Pool) -> ModelDeployRunnersRequest:
    return ModelDeployRunnersRequest(
        host=pool.host,
        host_dir="/srv/runners",
        compose_file=str(ROOT / "docker" / pool.compose),
        prefix=pool.prefix,
        count=pool.count,
        secondary=True,
        fleet_config=str(FLEET),
    )


def _expected_caches(pool: Pool) -> dict[str, tuple[int, str]]:
    return {
        pool.cache_base.format(n=n, name=name): (1001, "755")
        for n in range(1, pool.count + 1)
        for name in CACHE_NAMES
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("pool", [OMNIBASE_POOL, SCRATCH_POOL], ids=lambda p: p.prefix)
async def test_runner_cache_dirs_are_prepared_runner_owned_with_no_declared_job_root(
    pool: Pool,
) -> None:
    """Every ``~/.npm`` and ``~/.cache/*`` bind source is created as uid 1001 mode 755, read back, and reported."""
    handler = FakeHostHandler(pool)
    result = await handler.handle(_request(pool))

    assert result.success is True, result.error_message
    expected = _expected_caches(pool)
    assert expected.items() <= handler.dirs.items()
    assert set(expected) <= set(result.prepared)


@pytest.mark.asyncio
async def test_runner_cache_dirs_leave_the_workspace_and_foreign_binds_alone() -> None:
    """Only TMPDIR and cache sources are prepared: not the workspace, the hook, archive, docker socket or creds."""
    handler = FakeHostHandler(OMNIBASE_POOL)
    await handler.handle(_request(OMNIBASE_POOL))

    caches = set(_expected_caches(OMNIBASE_POOL))
    tmpdirs = {
        f"/mnt/h051-nvme/runners/hp051-omnibase-ci-runner/{n}/{SCRATCH}"
        for n in range(1, 17)
    }
    assert set(handler.dirs) - caches - tmpdirs <= {
        # the disk-admission pause directory is the one other source the effect has always prepared
        path
        for path in handler.dirs
        if path.endswith(("/.onex-disk-admission-pause", "/disk-admission-pause"))
    }
    assert not any(path.endswith("/work") for path in handler.dirs)


@pytest.mark.asyncio
async def test_runner_cache_dirs_a_root_owned_readback_refuses_the_install() -> None:
    """A cache directory that reads back root-owned after the install is a named refusal, not a silent pass."""
    victim = "/mnt/h051-nvme/runners/hp051-omnibase-ci-runner/9/cache/npm"
    handler = FakeHostHandler(OMNIBASE_POOL, root_owned=frozenset({victim}))
    result = await handler.handle(_request(OMNIBASE_POOL))

    assert result.success is False
    assert result.error_message is not None
    assert "RUNNER-TMP-VERIFY" in result.error_message
    assert victim in result.error_message
