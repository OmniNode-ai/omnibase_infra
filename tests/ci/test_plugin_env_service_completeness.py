# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Guardrail: every *_HOST env var in x-runtime-env whose default is a Docker
service name must have a corresponding service defined in docker-compose.infra.yml.

Catches: GAP-001 class bugs where a plugin is wired to a service that doesn't
exist in the compose network (e.g., OMNIMEMORY_MEMGRAPH_HOST defaults to
'omnibase-infra-memgraph' but the service wasn't defined).

This test encodes the rule that caught the Memgraph network topology gap described
in OMN-4309: any *_HOST var defaulting to a Docker service name must have a
matching service entry, or the runtime container will fail to resolve it at startup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

COMPOSE_FILE = (
    Path(__file__).parent.parent.parent / "docker" / "docker-compose.infra.yml"
)


def _extract_default(value: str) -> str | None:
    """Extract the default from a ${VAR:-default} compose interpolation."""
    match = re.match(r"\$\{[^}]+:-([^}]+)\}", str(value))
    return match.group(1) if match else None


def _looks_like_docker_service_name(value: str) -> bool:
    """True if value looks like a Docker service name (no dots, not localhost/127.x)."""
    if not value:
        return False
    if value in ("localhost", "127.0.0.1", "0.0.0.0"):  # noqa: S104
        return False
    if "." in value or ":" in value or "/" in value:
        return False
    return True


@pytest.mark.unit
def test_plugin_host_envvars_have_compose_services() -> None:
    """Every *_HOST var in x-runtime-env pointing at a Docker service must exist.

    This is the canonical regression test for GAP-001: a plugin configured with
    a HOST env var that defaults to a Docker service name that was never defined
    in the compose file. The runtime container cannot reach the service because
    Docker's internal DNS only resolves names that are declared as services in
    the same network.
    """
    compose = yaml.safe_load(COMPOSE_FILE.read_text())
    runtime_env: dict = compose.get("x-runtime-env", {})
    services: set[str] = set(compose.get("services", {}).keys())

    failures: list[str] = []
    for key, value in runtime_env.items():
        if not key.endswith("_HOST"):
            continue
        default = _extract_default(str(value))
        if default is None:
            continue
        if not _looks_like_docker_service_name(default):
            continue
        if default not in services:
            failures.append(
                f"  {key} defaults to '{default}' but no service '{default}' "
                f"is defined in docker-compose.infra.yml — runtime container cannot resolve it"
            )

    assert not failures, (
        "Plugin HOST env vars point at non-existent compose services:\n"
        + "\n".join(failures)
    )
