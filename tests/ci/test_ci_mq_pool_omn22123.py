# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22123: the reserved merge-queue label ``omnibase-ci-mq`` has declared runners on h051 and h053.

A merge_group job of a private repository asks for ``[self-hosted, omnibase-ci-mq]``. The runners that carried the label
lived on a host now in repair and the fleet declared none, so each job waited 14 to 60 minutes. This file holds what the
declaration must keep true: four runners per host, the reserved label alone (so no pull-request job can take them), a
runner and scratch name of their own, and no change to the ``action`` count the router's floor reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)
from tests.ci._runner_compose_text import resolve_runner_compose_defaults

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
COUNT = 4
RESERVED = {"self-hosted", "omnibase-ci-mq", "linux", "x64", "arch-amd64"}
# host -> (general pool prefix on that host, merge-queue pool prefix, action runners the host declared before OMN-22123)
POOLS = {
    "h051": ("hp051-ci-runner", "hp051-ci-mq-runner", 0),
    "h053": ("hp053-ci-runner", "hp053-ci-mq-runner", 8),
}


def _compose(prefix: str) -> dict[str, Any]:
    path = DOCKER / f"docker-compose.runners-{prefix}.yml"
    return cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("host", sorted(POOLS))
def test_every_merge_queue_runner_carries_the_reserved_label_and_no_shared_one(
    host: str,
) -> None:
    _, prefix, _ = POOLS[host]
    services = _compose(prefix)["services"]
    assert sorted(services) == [f"{prefix}-{n}" for n in range(1, COUNT + 1)]
    for name, service in services.items():
        labels = set(str(service["environment"]["RUNNER_LABELS"]).split(","))
        assert labels == RESERVED, name
        assert service["environment"]["RUNNER_GROUP"] == "omnibase-ci"
        assert service["environment"]["ONEX_LANE_HOST"] == host
        assert service["environment"]["RUNNER_NAME"] == name
        assert service["container_name"] == name
        assert f"{name}-creds:" in "".join(service["volumes"][-1:])


@pytest.mark.parametrize("host", sorted(POOLS))
def test_every_named_volume_a_runner_mounts_is_declared_at_the_top_level(
    host: str,
) -> None:
    """Compose refuses a project whose service mounts a named volume the file never declares."""
    _, prefix, _ = POOLS[host]
    document = _compose(prefix)
    declared = set(document.get("volumes", {}))
    mounted = {
        str(v).split(":", 1)[0]
        for s in document["services"].values()
        for v in s["volumes"]
        if isinstance(v, str) and not str(v).startswith(("/", "."))
    }
    assert mounted == {f"{prefix}-{n}-creds" for n in range(1, COUNT + 1)}
    assert mounted <= declared


@pytest.mark.parametrize("host", sorted(POOLS))
def test_every_runner_reads_the_host_timezone_database_read_only(host: str) -> None:
    """The runner image has no tzdata and the merge-group tests resolve ``America/New_York``."""
    _, prefix, _ = POOLS[host]
    for name, service in _compose(prefix)["services"].items():
        mounts = [
            v
            for v in service["volumes"]
            if isinstance(v, dict) and v.get("target") == "/usr/share/zoneinfo"
        ]
        assert len(mounts) == 1, name
        assert mounts[0]["source"] == "/usr/share/zoneinfo", name
        assert mounts[0]["read_only"] is True, name


# Hosts whose merge-queue runners keep their job directories on disk under /data/scratch. h051 is not one: see below.
DISK_SCRATCH_HOSTS = ("h053",)
MEMORY_HOSTS = ("h051",)
RUNNER_HOME = "/home/runner"
RUNNER_TMPDIR_NAME = "tmp"  # the runner's TMPDIR, a directory of its home
# What a runner keeps in memory: the job workspace, TMPDIR and the six package caches (OMN-22123).
IN_MEMORY_TARGETS = {
    f"{RUNNER_HOME}/actions-runner/_work",
    f"{RUNNER_HOME}/{RUNNER_TMPDIR_NAME}",
    f"{RUNNER_HOME}/.cache/pip",
    f"{RUNNER_HOME}/.cache/pre-commit",
    f"{RUNNER_HOME}/.cache/pnpm",
    f"{RUNNER_HOME}/.cache/ms-playwright",
    f"{RUNNER_HOME}/.cache/omni/ci-envs",
    f"{RUNNER_HOME}/.npm",
}
# The only mounts, besides the tmpfs ones, a memory-host runner may have: the socket, its own hooks, the pause directory, the credentials, the
# host table and the timezone database. None is a job directory or a cache.
ALLOWED_BIND_TARGETS = {
    "/var/run/docker.sock",
    "/opt/onex-runner-hooks",
    f"{RUNNER_HOME}/.onex-disk-admission-pause",
    f"{RUNNER_HOME}/.lab-credentials",
    "/etc/onex/lab_run_hosts.yaml",
    "/usr/share/zoneinfo",
    f"{RUNNER_HOME}/.runner-creds",
}


def _bytes(limit: str) -> int:
    units = {"g": 1 << 30, "m": 1 << 20}
    return int(str(limit)[:-1]) * units[str(limit)[-1].lower()]


def _target(volume: Any) -> str:
    """The container path of a short or long volume entry, with the runner home resolved."""
    if isinstance(volume, dict):
        return resolve_runner_compose_defaults(str(volume["target"]))
    return resolve_runner_compose_defaults(str(volume)).split(":")[1]


@pytest.mark.parametrize("host", DISK_SCRATCH_HOSTS)
def test_scratch_paths_never_alias_the_general_pool_on_the_same_host(
    host: str,
) -> None:
    general, prefix, _ = POOLS[host]
    mq_paths = {
        str(v).split(":", 1)[0]
        for s in _compose(prefix)["services"].values()
        for v in s["volumes"]
        if str(v).startswith("/data/scratch/")
    }
    general_paths = {
        str(v).split(":", 1)[0]
        for s in _compose(general)["services"].values()
        for v in s["volumes"]
        if str(v).startswith("/data/scratch/")
    }
    assert mq_paths and not (mq_paths & general_paths)
    assert all("/mq-" in p for p in mq_paths)


@pytest.mark.parametrize("host", MEMORY_HOSTS)
def test_every_merge_queue_job_directory_is_a_tmpfs_and_no_disk_path_is_bound(
    host: str,
) -> None:
    """OMN-22123, operator RULING 2026-11-04T07:03:44Z: the queue runners run on the memory drive, not the data disk.

    Work, TMPDIR and the six caches are tmpfs mounts of the container, each with a size ceiling, so a recreate cannot
    bring a disk directory back and no host unit has to recreate a path after a reboot. Every other bind is one the
    runner needs and none of them is a job directory or a cache.
    """
    _, prefix, _ = POOLS[host]
    for name, service in _compose(prefix)["services"].items():
        tmpfs = {
            _target(v): v
            for v in service["volumes"]
            if isinstance(v, dict) and v.get("type") == "tmpfs"
        }
        assert set(tmpfs) == IN_MEMORY_TARGETS, name
        for target, mount in tmpfs.items():
            assert mount["tmpfs"]["size"], (name, target)
        binds = {
            _target(v)
            for v in service["volumes"]
            if not (isinstance(v, dict) and v.get("type") == "tmpfs")
        }
        assert binds <= ALLOWED_BIND_TARGETS, (name, binds - ALLOWED_BIND_TARGETS)
        text = repr(service["volumes"])
        assert "/data/scratch" not in text and "/mq-" not in text, name


@pytest.mark.parametrize("host", MEMORY_HOSTS)
def test_the_memory_limit_covers_the_tmpfs_ceilings_it_is_charged_for(
    host: str,
) -> None:
    """A tmpfs page is charged to the container, so the limit must hold the workspace beside the job's own memory.

    Measured 2026-11-04 on h051: a shard holds 3.2 to 3.5 GiB in its job workspace and the old 4 GiB limit was reached
    4400 to 6200 times per runner. The limit is the general pool's 6 GiB plus the workspace, and swap equals the limit so
    the in-memory directories never reach the disk.
    """
    _, prefix, _ = POOLS[host]
    for name, service in _compose(prefix)["services"].items():
        limit = _bytes(service["mem_limit"])
        assert limit >= 10 << 30, name
        assert _bytes(service["memswap_limit"]) == limit, name
        work = next(
            v
            for v in service["volumes"]
            if isinstance(v, dict) and _target(v).endswith("/_work")
        )
        assert _bytes(str(work["tmpfs"]["size"]).replace("gb", "g")) < limit, name


def test_the_fleet_declares_four_merge_queue_runners_on_each_host() -> None:
    config = load_runner_fleet_config(FLEET)
    for host_name, (_, prefix, _) in POOLS.items():
        host = config.host_of_runner(f"{prefix}-{COUNT}")
        assert host is not None and host.host == host_name
        assert host.declared_count("merge-queue") == COUNT
    document = yaml.safe_load(FLEET.read_text(encoding="utf-8"))
    declared = [
        pool
        for row in document["hosts"]
        for pool in row.get("pools", [])
        if "merge-queue" in pool["classes"]
    ]
    assert sorted(p["runner_name_prefix"] for p in declared) == sorted(
        prefix for _, prefix, _ in POOLS.values()
    )
    assert all(p["classes"] == ["merge-queue"] for p in declared)


def test_the_merge_queue_pools_move_no_other_class_count() -> None:
    """``merge-queue`` is its own class, so the ``action`` floor and the verify counts read what they read before."""
    config = load_runner_fleet_config(FLEET)
    for host_name, (_, _, action) in POOLS.items():
        host = next(h for h in config.hosts if h.host == host_name)
        assert host.declared_count("merge-queue") == COUNT
        assert host.declared_count("action") == action
        assert host.declared_count("verify") == 0
