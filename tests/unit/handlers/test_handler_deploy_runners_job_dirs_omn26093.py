# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26093: the prepare effect creates every bind under the host's declared job-directory root, runner-owned.

The h053 pool binds its workspace, TMPDIR and six caches under ``job_dir_host_root`` of its host row. Docker creates a
missing bind source root-owned, and the entrypoint hands only ``_work`` and ``~/.cache`` itself to the runner, so a
``cache/pip`` source created by Docker would be unwritable for the job user. The effect creates them first, TMPDIR mode
1777 and the rest mode 755, and reads each back.
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

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / "docker" / "docker-compose.runners-hp053-ci-runner.yml"
FLEET = ROOT / "config" / "runner_fleet.yaml"
PREFIX = "hp053-ci-runner"
NVME_ROOT = "/mnt/h053-nvme/runners/hp053-ci-runner"
HOME = "/home/runner"
SCRATCH = "tmp"
POOL_COUNT = 5


def _rendered_compose() -> str:
    """What ``docker compose config --format json`` prints for the shipped pool: long-form binds, home expanded."""
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services: dict[str, Any] = {}
    for name, service in document["services"].items():
        volumes = []
        for volume in service["volumes"]:
            if isinstance(volume, str) and volume.startswith("/"):
                source, _, target = volume.partition(":")
                volumes.append(
                    {
                        "type": "bind",
                        "source": source,
                        "target": target.replace(
                            "${RUNNER_USER_HOME:-/home/runner}", HOME
                        ),
                    }
                )
        services[name] = {
            "environment": {"TMPDIR": f"{HOME}/{SCRATCH}"},
            "volumes": volumes,
        }
    return json.dumps({"services": services})


class FakeHostHandler(HandlerDeployRunners):
    """ssh replaced by an in-memory host: ``install -d`` records owner and mode, ``stat`` reads them back."""

    def __init__(self, *, job_root_declared: bool = True) -> None:
        self.dirs: dict[str, tuple[int, str]] = {}
        self._rendered = _rendered_compose()
        self._job_root_declared = job_root_declared

    async def _converge_lab_host_table(
        self, request: ModelDeployRunnersRequest, timeout: int
    ) -> str | None:
        return None  # OMN-26098: covered by test_handler_deploy_runners_lab_host_table_omn26098

    async def _remote(self, host: str, command: str, timeout: int) -> str:
        if "compose" in command and "config" in command:
            return self._rendered
        if " install " in f" {command} ":
            tokens = shlex.split(command.split("; ")[-1])
            owner = int(tokens[tokens.index("-o") + 1])
            mode = tokens[tokens.index("-m") + 1]
            self.dirs[tokens[-1]] = (owner, mode)
            return ""
        match = re.search(r"stat -c '%F:%u:%a' -- (\S+)", command)
        assert match is not None, command
        owner, mode = self.dirs[shlex.split(match.group(1))[0]]
        return f"directory:{owner}:{mode}\n"


def _request(fleet: Path = FLEET) -> ModelDeployRunnersRequest:
    return ModelDeployRunnersRequest(
        host="h053",
        host_dir="/srv/runners",
        compose_file=str(COMPOSE),
        prefix=PREFIX,
        count=POOL_COUNT,
        secondary=True,
        fleet_config=str(fleet),
    )


@pytest.mark.asyncio
async def test_every_job_directory_is_created_runner_owned_under_the_declared_root() -> (
    None
):
    handler = FakeHostHandler()
    result = await handler.handle(_request())

    assert result.success is True, result.error_message
    expected: dict[str, tuple[int, str]] = {}
    for number in range(1, POOL_COUNT + 1):
        base = f"{NVME_ROOT}/{number}"
        expected[f"{base}/{SCRATCH}"] = (1001, "1777")
        expected[f"{base}/work"] = (1001, "755")
        for cache in ("pip", "pre-commit", "pnpm", "ms-playwright", "ci-envs", "npm"):
            expected[f"{base}/cache/{cache}"] = (1001, "755")
    assert handler.dirs == expected
    assert sorted(result.prepared) == sorted(expected)


@pytest.mark.asyncio
async def test_a_pool_with_no_declared_root_prepares_tmpdir_and_caches_but_not_the_workspace(
    tmp_path: Path,
) -> None:
    """Control: without ``job_dir_host_root`` the workspace is not touched; OMN-26113 prepares the cache binds regardless."""
    document = yaml.safe_load(FLEET.read_text(encoding="utf-8"))
    for row in document["hosts"]:
        row.pop("job_dir_host_root", None)
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text(yaml.dump(document))
    handler = FakeHostHandler()
    result = await handler.handle(_request(fleet))

    assert result.success is True, result.error_message
    assert sorted(handler.dirs) == sorted(
        [f"{NVME_ROOT}/{n}/{SCRATCH}" for n in range(1, POOL_COUNT + 1)]
        + [
            f"{NVME_ROOT}/{n}/cache/{cache}"
            for n in range(1, POOL_COUNT + 1)
            for cache in (
                "pip",
                "pre-commit",
                "pnpm",
                "ms-playwright",
                "ci-envs",
                "npm",
            )
        ]
    )


@pytest.mark.asyncio
async def test_a_source_outside_the_root_is_not_prepared() -> None:
    """The root bounds the effect: the host timezone, hook and archive binds are never created or chowned."""
    handler = FakeHostHandler()
    await handler.handle(_request())
    assert all(path.startswith(NVME_ROOT + "/") for path in handler.dirs)
