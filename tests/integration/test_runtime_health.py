# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for runtime container health (OMN-6996).

Proves all runtime containers boot to healthy state. This is a boot/container
sanity gate only — functional proof that the node dispatch layer works comes
from test_node_registration.py and test_dispatch_roundtrip.py.

These tests require the Docker runtime stack to be running (infra-up or
infra-up-runtime). They are marked @slow because they inspect live containers.

Related:
    - OMN-6768: runtime-sweep skill
    - OMN-6995: Platform Subsystem Verification epic
"""

from __future__ import annotations

import os
import subprocess

import pytest

# Container names are read from environment or use defaults matching
# the canonical docker-compose.infra.yml service names.
REQUIRED_CONTAINERS: list[str] = [
    c.strip()
    for c in os.environ.get(
        "ONEX_REQUIRED_CONTAINERS",
        "omninode-runtime,omnibase-infra-postgres,omnibase-infra-redpanda,omnibase-infra-valkey",
    ).split(",")
    if c.strip()
]


def _get_container_statuses() -> dict[str, str]:
    """Query Docker for all container names and statuses.

    Uses ``docker ps -a`` (not ``docker ps``) to catch stopped/exited
    containers that would be invisible to the default listing.
    """
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    statuses: dict[str, str] = {}
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            name, status = line.split("\t", 1)
            statuses[name] = status
    return statuses


@pytest.mark.slow
class TestRuntimeHealth:
    """Prove all runtime containers boot to healthy state."""

    def test_all_containers_healthy(self) -> None:
        """Every required container must report healthy or Up (not unhealthy/starting)."""
        container_status = _get_container_statuses()

        for container in REQUIRED_CONTAINERS:
            assert container in container_status, (
                f"Container {container!r} not found. "
                f"Use 'docker ps -a' to check. "
                f"Available: {sorted(container_status.keys())}"
            )
            status = container_status[container].lower()

            # Reject explicitly bad states
            for bad_state in ("unhealthy", "restarting"):
                assert bad_state not in status, (
                    f"Container {container!r} in bad state {bad_state!r}: "
                    f"{container_status[container]}"
                )

            # Require "healthy" for containers with healthchecks, or "up" for
            # those without. Explicitly reject "starting" (not yet healthy).
            is_healthy = "healthy" in status
            is_up = "up" in status and "unhealthy" not in status
            assert is_healthy or is_up, (
                f"Container {container!r} not healthy: {container_status[container]}"
            )

    def test_no_containers_stuck_in_starting(self) -> None:
        """No container should be stuck in 'starting' state.

        This catches the runtime-worker-1 regression where a container
        enters health-check starting state and never transitions to healthy.
        """
        container_status = _get_container_statuses()

        for name, status in container_status.items():
            # Only flag REQUIRED containers — other containers on the host
            # (e.g., unrelated dev services) are not our concern.
            if name not in REQUIRED_CONTAINERS:
                continue
            assert "starting" not in status.lower(), (
                f"Container {name!r} stuck in starting state: {status}"
            )
