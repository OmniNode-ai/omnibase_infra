# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26093: the h053 general pool keeps its job directories on the NVMe, bound under the fleet's declared root.

OMN-19998 put them in tmpfs (operator RULING 2026-11-04T07:03:44Z: not on a spinning disk). On 2026-11-19 the five live
containers still bound /data/scratch on h053's rotational, 93 percent full root volume (io pressure full avg60 35 percent,
96 ms write wait), and Repo Scripts Tests ran 10.1 to 10.3 minutes against a 10 minute budget. The directories now sit under
``job_dir_host_root`` of the host's row in config/runner_fleet.yaml, on /mnt/h053-nvme (a non-rotational device), one
``<N>/`` directory per runner. Nothing of the pool is tmpfs, so nothing of it is charged to the runner's memory limit: the
limit is 6 GiB, the value the live runners ran seven days with oom_kill 0 and 0.1 to 0.66 GiB of anonymous memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners-hp053-ci-runner.yml"
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
RUNNER_HOME = "${RUNNER_USER_HOME:-/home/runner}"
PREFIX = "hp053-ci-runner"
SCRATCH = "tmp"
NVME_ROOT = "/mnt/h053-nvme/runners/hp053-ci-runner"
# container target -> source relative to <root>/<N>
JOB_DIRECTORIES = {
    f"{RUNNER_HOME}/actions-runner/_work": "work",
    f"{RUNNER_HOME}/{SCRATCH}": SCRATCH,
    f"{RUNNER_HOME}/.cache/pip": "cache/pip",
    f"{RUNNER_HOME}/.cache/pre-commit": "cache/pre-commit",
    f"{RUNNER_HOME}/.cache/pnpm": "cache/pnpm",
    f"{RUNNER_HOME}/.cache/ms-playwright": "cache/ms-playwright",
    f"{RUNNER_HOME}/.cache/omni/ci-envs": "cache/ci-envs",
    f"{RUNNER_HOME}/.npm": "cache/npm",
}
GIB = 1024**3
SLICE = (
    REPO_ROOT
    / "docker"
    / "runners"
    / "systemd"
    / "pools"
    / "hp053-ci-runner"
    / "omnirunners.slice"
)


def _services() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return cast("dict[str, dict[str, Any]]", document["services"])


def _binds(service: dict[str, Any]) -> dict[str, str]:
    """Short-form ``source:target`` volumes of the service, as target -> source."""
    result: dict[str, str] = {}
    for volume in service["volumes"]:
        if isinstance(volume, str) and volume.startswith("/"):
            source, _, target = volume.partition(":")
            result[target] = source
    return result


def test_the_fleet_declares_the_nvme_root_for_the_pool() -> None:
    host = next(h for h in load_runner_fleet_config(FLEET).hosts if h.host == "h053")
    assert host.runner_name_prefix == PREFIX
    assert host.job_dir_host_root == NVME_ROOT


def test_every_job_directory_is_a_bind_under_the_declared_nvme_root_and_none_is_tmpfs() -> (
    None
):
    services = _services()
    assert len(services) == 5
    for name, service in services.items():
        number = name.rsplit("-", 1)[1]
        assert "tmpfs" not in service, name
        binds = _binds(service)
        for target, relative in JOB_DIRECTORIES.items():
            assert binds[target] == f"{NVME_ROOT}/{number}/{relative}", (name, target)
        # Nothing of the job path sits on the root volume: the old /data/scratch binds are gone.
        for volume in service["volumes"]:
            assert "/data/scratch" not in repr(volume), (name, volume)


def test_the_temp_directory_is_the_declared_bind() -> None:
    for name, service in _services().items():
        number = name.rsplit("-", 1)[1]
        assert service["environment"]["TMPDIR"] == f"{RUNNER_HOME}/{SCRATCH}", name
        assert (
            _binds(service)[f"{RUNNER_HOME}/{SCRATCH}"]
            == f"{NVME_ROOT}/{number}/{SCRATCH}"
        )


def _slice_limit(key: str) -> int:
    for line in SLICE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            value = line.split("=", 1)[1].strip()
            assert value.endswith("G"), line
            return int(value[:-1]) * GIB
    raise AssertionError(f"{key} not set in {SLICE}")


def test_the_limits_are_the_ones_the_host_memory_reads_allow() -> None:
    services = _services()
    assert len(services) == 5
    for name, service in services.items():
        assert service["mem_limit"] == service["memswap_limit"] == "6g", name
        assert float(service["cpus"]) == 5.0, name
    total = sum(_size_bytes_of_limit(s["mem_limit"]) for s in services.values())
    # The slice read from disk must admit every runner at its limit, and cap it with headroom for job containers.
    assert total <= _slice_limit("MemoryHigh") <= _slice_limit("MemoryMax")
    assert _slice_limit("MemoryMax") - total >= 1 * GIB
    # OMN-26093: the pool shares the slice with the four merge-queue runners (4 GiB each, read 2026-11-19), so all nine
    # at their limits are 46 GiB, which the slice's MemoryMax admits exactly.
    assert total + 4 * 4 * GIB <= _slice_limit("MemoryMax")
    # The slice cap stays under the 49 GiB the host showed as its lowest available memory on 2026-11-04.
    assert _slice_limit("MemoryMax") <= 49 * GIB


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
