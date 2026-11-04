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
