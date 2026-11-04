# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: the h053 general pool keeps its job directories in memory and is sized from the host's measured memory.

Operator RULING 2026-11-04T07:03:44Z: runners go on a memory drive, not spinning disk. h053's root volume is a rotational
disk, and 49 to 55 GiB of its 125 GiB were available on 2026-11-04 with the lab-work units running, so the pool is five
runners at 9 GiB (memswap equal, nothing swaps) under a slice that caps all five at 46 GiB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners-hp053-ci-runner.yml"
RUNNER_HOME = "${RUNNER_USER_HOME:-/home/runner}"
SCRATCH = "tmp"
IN_MEMORY_DIRS = (
    f"{RUNNER_HOME}/actions-runner/_work",
    f"{RUNNER_HOME}/{SCRATCH}",
    f"{RUNNER_HOME}/.cache/pip",
    f"{RUNNER_HOME}/.cache/pre-commit",
    f"{RUNNER_HOME}/.cache/pnpm",
    f"{RUNNER_HOME}/.cache/ms-playwright",
    f"{RUNNER_HOME}/.cache/omni/ci-envs",
    f"{RUNNER_HOME}/.npm",
)
GIB = 1024**3


def _services() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return cast("dict[str, dict[str, Any]]", document["services"])


def _size_bytes(entry: str) -> int:
    size = next(
        option.split("=", 1)[1]
        for option in entry.rsplit(":", 1)[1].split(",")
        if option.startswith("size=")
    )
    assert size.endswith("g"), entry
    return int(size[:-1]) * GIB


def test_every_job_directory_is_an_exec_tmpfs_and_no_disk_path_is_bound() -> None:
    for name, service in _services().items():
        entries = service["tmpfs"]
        assert [e.rsplit(":", 1)[0] for e in entries] == list(IN_MEMORY_DIRS), name
        for entry in entries:
            options = entry.rsplit(":", 1)[1].split(",")
            assert "exec" in options and "mode=1777" in options, (name, entry)
            assert not any(option.startswith("no") for option in options), entry
        for volume in service["volumes"]:
            assert "/data/scratch" not in repr(volume), (name, volume)


def test_the_job_directories_leave_the_free_space_the_lane_probe_requires() -> None:
    for name, service in _services().items():
        sizes = {e.rsplit(":", 1)[0]: _size_bytes(e) for e in service["tmpfs"]}
        assert sizes[IN_MEMORY_DIRS[0]] >= 12 * GIB, name
        assert sizes[IN_MEMORY_DIRS[1]] >= 12 * GIB, name


def test_the_limits_are_the_ones_the_host_memory_reads_allow() -> None:
    services = _services()
    assert len(services) == 5
    for name, service in services.items():
        assert service["mem_limit"] == service["memswap_limit"] == "9g", name
        assert float(service["cpus"]) == 5.0, name
    # Five runners at 9 GiB are 45 GiB: under the 49 GiB the host showed as its lowest available memory on 2026-11-04.
    assert (
        sum(_size_bytes_of_limit(s["mem_limit"]) for s in services.values()) <= 49 * GIB
    )


def _size_bytes_of_limit(value: str) -> int:
    assert value.endswith("g"), value
    return int(value[:-1]) * GIB


def test_every_runner_reads_the_host_timezone_database_read_only() -> None:
    for name, service in _services().items():
        zoneinfo = [
            v
            for v in service["volumes"]
            if isinstance(v, dict) and v.get("target") == "/usr/share/zoneinfo"
        ]
        assert len(zoneinfo) == 1 and zoneinfo[0]["read_only"] is True, name
