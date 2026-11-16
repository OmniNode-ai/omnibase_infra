# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the runner compose file with its defaulted path variables resolved.

``docker/docker-compose.runners.yml`` spells the runner user's home and the
runner host's deploy home as ``${VAR:-default}`` so the file holds no
machine-specific path. Compose resolves them to the default when the variable is
unset, which is how every deploy runs. Tests that assert on the mount targets
and credential paths the fleet actually gets read the file through this module.
"""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

RUNNER_HOME = "/home/runner"
RUNNER_HOST_HOME = "~"
RUNNER_HOME_VARIABLE = "${RUNNER_USER_HOME:-" + RUNNER_HOME + "}"


def resolve_runner_compose_defaults(text: str) -> str:
    """Replace the defaulted runner-home variable with its default."""
    return text.replace(RUNNER_HOME_VARIABLE, RUNNER_HOME)


def read_runner_compose(path: Path) -> str:
    """The compose file's text as compose renders it with no variable set."""
    return resolve_runner_compose_defaults(path.read_text(encoding="utf-8"))


def general_pool_control() -> tuple[str, Path]:
    """The first runner of the largest general pool off the primary host, and its compose file.

    OMN-22898 retired the primary host's own general pool, so the positive controls that
    ask "does a general-pool runner carry this wiring" read the largest pool of
    `action` runners declared on another host, named from the fleet config.
    """
    repo_root = Path(__file__).resolve().parents[2]
    config = load_runner_fleet_config(repo_root / "config" / "runner_fleet.yaml")
    pools = [
        pool
        for host in config.hosts
        if host.host != config.runner_host
        for pool in host.pools
        if "action" in pool.classes
    ]
    pool = max(pools, key=lambda candidate: candidate.expected_count)
    prefix = pool.runner_name_prefix
    return f"{prefix}-1", repo_root / "docker" / f"docker-compose.runners-{prefix}.yml"
