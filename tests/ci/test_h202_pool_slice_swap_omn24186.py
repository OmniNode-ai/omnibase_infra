# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24186: a CI pool that shares its host with the model server declares MemorySwapMax=0.

The fleet inventory (``config/runner_fleet.yaml``) gives one host a ``pools:`` list: its CI pool runs beside the
model server and the lab lanes, inside its own aggregate cgroup (``docker/runners/systemd/pools/<prefix>/
omnirunners.slice``). That slice used to declare ``MemorySwapMax=2G``. The host's swap file sits on the same SATA SSD
as the runners' scratch directories. Read on 2026-11-13, 22 hours after the host booted: the slice hit its 2G cap
323,840 times, its swap peak equalled the cap, and it accounted for 6,531,559 of the host's 14,348,433 swap-in pages
(45 percent) and 9,479,792 of its 30,216,307 swap-out pages (31 percent); the lane admission refused a lane at
11:49Z on swap-in 43,800 KB/s against its 20,000 bar. The hp053 pool already declares ``MemorySwapMax=0``
(``tests/ci/test_hp053_pool_omn22111.py``); this file holds the same rule for the shared-host pool, and the
ceilings it keeps (``MemoryHigh`` 20G, ``MemoryMax`` 24G) stay where they were.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
POOLS = DOCKER / "runners" / "systemd" / "pools"
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
SLICE_NAME = "omnirunners.slice"
MEASURED = (
    "OMN-24186: read on 2026-11-13 the slice hit its 2G swap cap 323,840 times in 22 hours, its swap peak "
    "equalled the cap, and it accounted for 6,531,559 of the host's 14,348,433 swap-in pages (45 percent) "
    "since boot; the lane admission refused a lane at 11:49Z on swap-in 43,800 KB/s against its 20,000 bar"
)


def _bytes(value: str) -> int:
    assert value[-1].lower() == "g", value
    return int(value[:-1]) * 1024**3


def _limits(unit: Path) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for key, value in (
            line.split("=", 1)
            for line in unit.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.lstrip().startswith(("#", "["))
        )
    }


def _shared_host_pool_prefixes() -> list[str]:
    """Every pool the inventory lists under a host's ``pools:`` key that ships its own slice file."""
    document = cast("dict[str, Any]", yaml.safe_load(FLEET.read_text(encoding="utf-8")))
    prefixes = [
        str(pool["runner_name_prefix"])
        for host in document["hosts"]
        for pool in host.get("pools", [])
    ]
    return [p for p in prefixes if (POOLS / p / SLICE_NAME).is_file()]


def test_the_inventory_names_a_shared_host_pool_with_its_own_slice() -> None:
    """Positive control: the loop below has something to judge, so it cannot pass on an empty list."""
    assert _shared_host_pool_prefixes(), (
        "no pool under a host's pools: key ships a slice file"
    )


@pytest.mark.parametrize("prefix", _shared_host_pool_prefixes())
def test_the_shared_host_pool_slice_allows_no_swap(prefix: str) -> None:
    limits = _limits(POOLS / prefix / SLICE_NAME)
    assert limits["MemorySwapMax"] == "0", f"{prefix}: {MEASURED}"


@pytest.mark.parametrize("prefix", _shared_host_pool_prefixes())
def test_the_shared_host_pool_slice_keeps_its_memory_ceilings(prefix: str) -> None:
    limits = _limits(POOLS / prefix / SLICE_NAME)
    assert limits["MemoryHigh"] == "20G", prefix
    assert limits["MemoryMax"] == "24G", prefix
    assert limits["CPUWeight"] == "25", prefix
    assert _bytes(limits["MemoryHigh"]) <= _bytes(limits["MemoryMax"]), prefix


@pytest.mark.parametrize("prefix", _shared_host_pool_prefixes())
def test_the_shared_host_pool_compose_points_at_the_slice(prefix: str) -> None:
    compose = DOCKER / f"docker-compose.runners-{prefix}.yml"
    document = cast(
        "dict[str, Any]", yaml.safe_load(compose.read_text(encoding="utf-8"))
    )
    assert document["services"], prefix
    for name, service in document["services"].items():
        assert service["cgroup_parent"] == SLICE_NAME, name
