# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker subnet-pool occupancy collector for a runner host.

Counts the Docker networks on a runner host and derives the address-pool
capacity from the daemon's ``default-address-pool`` configuration so the
pre-exhaustion threshold tracks the *actual* pool, not a hardcoded guess.
"""

from __future__ import annotations

import asyncio
import json

from omnibase_infra.observability.runner_health.model_network_pool_status import (
    ModelNetworkPoolStatus,
)

# Docker's compiled-in default when no ``default-address-pool`` is configured:
# the 172.17.0.0/16 .. 172.31.0.0/16 + 192.168.0.0/16 blocks subnetted at /24,
# which yields a few hundred usable subnets. We use a conservative fleet-tuned
# default that matches the observed `.201` exhaustion point (~31 networks before
# the leak floods the practical working set). Overridable via fleet config.
_FALLBACK_POOL_CAPACITY = 31


class CollectorNetworkPool:
    """Collects Docker subnet-pool occupancy from a runner host via SSH."""

    def __init__(
        self,
        runner_host: str,
        pool_capacity: int = _FALLBACK_POOL_CAPACITY,
        warn_threshold_ratio: float = 0.8,
    ) -> None:
        self._runner_host = runner_host
        self._pool_capacity = pool_capacity
        self._warn_threshold_ratio = warn_threshold_ratio

    async def _fetch_network_count(self) -> int:
        """Return the count of Docker networks on the host via SSH."""
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._runner_host,
            "docker network ls -q | wc -l",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return 0
        try:
            return int(stdout.decode().strip())
        except ValueError:
            return 0

    async def _fetch_pool_capacity(self) -> int:
        """Derive subnet-pool capacity from the daemon's default-address-pool.

        Falls back to the fleet-tuned capacity when the daemon does not expose
        a pool config (older daemons, or pool inspection failure).
        """
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            self._runner_host,
            "docker info --format '{{json .DefaultAddressPools}}' 2>/dev/null",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return self._pool_capacity
        capacity = _capacity_from_pool_json(stdout.decode().strip())
        return capacity if capacity > 0 else self._pool_capacity

    async def collect(self) -> ModelNetworkPoolStatus:
        """Collect a point-in-time subnet-pool occupancy snapshot."""
        count, capacity = await asyncio.gather(
            self._fetch_network_count(),
            self._fetch_pool_capacity(),
        )
        return ModelNetworkPoolStatus(
            host=self._runner_host,
            network_count=count,
            pool_capacity=capacity,
            warn_threshold_ratio=self._warn_threshold_ratio,
        )


def _capacity_from_pool_json(raw: str) -> int:
    """Compute total subnets available from a DefaultAddressPools JSON blob.

    Each pool entry is ``{"Base": "172.17.0.0/16", "Size": 24}``. The number
    of subnets a pool yields is ``2 ** (Size - base_prefix)``. Returns 0 when
    the blob is empty/unparseable so the caller can fall back.
    """
    if not raw or raw == "null":
        return 0
    try:
        pools = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if not isinstance(pools, list):
        return 0
    total = 0
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        base = pool.get("Base", "")
        size = pool.get("Size")
        if not isinstance(base, str) or "/" not in base or not isinstance(size, int):
            continue
        try:
            base_prefix = int(base.rsplit("/", 1)[1])
        except ValueError:
            continue
        if size < base_prefix:
            continue
        total += 2 ** (size - base_prefix)
    return total


__all__ = ["CollectorNetworkPool"]
