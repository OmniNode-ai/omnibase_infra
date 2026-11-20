# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26093: the shipped hp053 pool, its fleet row and the deploy effect agree on the NVMe job directories.

Three artifacts have to say the same thing for the pool's job workspace, TMPDIR and caches to land on the NVMe and
not on the rotational root volume: the compose file's bind sources, the ``job_dir_host_root`` of the host's row in
``config/runner_fleet.yaml`` (read through the real fleet loader), and what ``HandlerDeployRunners`` creates before
a container exists. Only the ssh transport is replaced; the compose, the fleet file and the handler are the shipped ones.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_deploy_runners import HandlerDeployRunners
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "docker" / "docker-compose.runners-hp053-ci-runner.yml"
FLEET = ROOT / "config" / "runner_fleet.yaml"
PREFIX = "hp053-ci-runner"
HOST = "h053"
POOL_COUNT = 5
RUNNER_HOME = "/home/runner"
SCRATCH = "tmp"
ROOT_VOLUME_PREFIXES = ("/data/scratch", "/var/lib/docker")
JOB_TARGET_SUFFIXES = (
    "/actions-runner/_work",
    f"/{SCRATCH}",
    "/.cache/pip",
    "/.cache/pre-commit",
    "/.cache/pnpm",
    "/.cache/ms-playwright",
    "/.cache/omni/ci-envs",
    "/.npm",
)


def _compose_services() -> dict[str, Any]:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services: dict[str, Any] = document["services"]
    return services


def _job_binds(service: dict[str, Any]) -> dict[str, str]:
    """Container target to host source for every short-form bind of a runner's job paths."""
    binds: dict[str, str] = {}
    for volume in service["volumes"]:
        if not isinstance(volume, str) or not volume.startswith("/"):
            continue
        source, _, target = volume.partition(":")
        target = target.replace("${RUNNER_USER_HOME:-/home/runner}", RUNNER_HOME)
        if target.endswith(JOB_TARGET_SUFFIXES) and target.startswith(RUNNER_HOME):
            binds[target] = source
    return binds


def _rendered(services: dict[str, Any]) -> str:
    rendered: dict[str, Any] = {}
    for name, service in services.items():
        volumes = []
        for volume in service["volumes"]:
            if isinstance(volume, str) and volume.startswith("/"):
                source, _, target = volume.partition(":")
                volumes.append(
                    {
                        "type": "bind",
                        "source": source,
                        "target": target.replace(
                            "${RUNNER_USER_HOME:-/home/runner}", RUNNER_HOME
                        ),
                    }
                )
        rendered[name] = {
            "environment": {"TMPDIR": f"{RUNNER_HOME}/{SCRATCH}"},
            "volumes": volumes,
        }
    return json.dumps({"services": rendered})


class RecordingHostHandler(HandlerDeployRunners):
    """The real handler with ssh replaced by a host that records what it was told to create."""

    def __init__(self, rendered: str) -> None:
        self.created: dict[str, tuple[int, str]] = {}
        self._rendered = rendered

    async def _converge_lab_host_table(
        self, request: ModelDeployRunnersRequest, timeout: int
    ) -> str | None:
        return None  # OMN-26098: covered by test_handler_deploy_runners_lab_host_table_omn26098

    async def _remote(self, host: str, command: str, timeout: int) -> str:
        if "compose" in command and "config" in command:
            return self._rendered
        if " install " in f" {command} ":
            tokens = shlex.split(command.split("; ")[-1])
            self.created[tokens[-1]] = (
                int(tokens[tokens.index("-o") + 1]),
                tokens[tokens.index("-m") + 1],
            )
            return ""
        match = re.search(r"stat -c '%F:%u:%a' -- (\S+)", command)
        assert match is not None, command
        owner, mode = self.created[shlex.split(match.group(1))[0]]
        return f"directory:{owner}:{mode}\n"


def _declared_root() -> str:
    row = next(
        h
        for h in load_runner_fleet_config(FLEET).hosts
        if h.host == HOST and h.runner_name_prefix == PREFIX
    )
    assert row.job_dir_host_root is not None
    return row.job_dir_host_root


def test_every_job_path_of_every_runner_binds_under_the_fleet_rows_root() -> None:
    root = _declared_root()
    services = _compose_services()
    assert sorted(services) == [f"{PREFIX}-{n}" for n in range(1, POOL_COUNT + 1)]
    for number in range(1, POOL_COUNT + 1):
        binds = _job_binds(services[f"{PREFIX}-{number}"])
        assert set(binds) == {
            f"{RUNNER_HOME}{suffix}" for suffix in JOB_TARGET_SUFFIXES
        }, binds
        for target, source in binds.items():
            assert source.startswith(f"{root}/{number}/"), (target, source)
            assert not source.startswith(ROOT_VOLUME_PREFIXES), (target, source)
    assert not any(
        isinstance(volume, dict) and volume.get("type") == "tmpfs"
        for service in services.values()
        for volume in service["volumes"]
    )
    assert "tmpfs" not in {key for s in services.values() for key in s}


@pytest.mark.asyncio
async def test_the_deploy_effect_prepares_exactly_the_compose_job_binds() -> None:
    services = _compose_services()
    handler = RecordingHostHandler(_rendered(services))
    result = await handler.handle(
        ModelDeployRunnersRequest(
            host=HOST,
            host_dir="/srv/runners",
            compose_file=str(COMPOSE),
            prefix=PREFIX,
            count=POOL_COUNT,
            secondary=True,
            fleet_config=str(FLEET),
        )
    )

    assert result.success is True, result.error_message
    declared = {
        source
        for number in range(1, POOL_COUNT + 1)
        for source in _job_binds(services[f"{PREFIX}-{number}"]).values()
    }
    assert set(handler.created) == declared
    assert sorted(result.prepared) == sorted(declared)
    assert all(owner == 1001 for owner, _ in handler.created.values())
    assert all(path.startswith(_declared_root() + "/") for path in handler.created)
