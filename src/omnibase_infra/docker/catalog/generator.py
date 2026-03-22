# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose generator — renders a ResolvedStack into docker-compose YAML."""

from __future__ import annotations

import re

# Compose YAML values are heterogeneous dicts. We use dict[str, object]
# instead of dict[str, Any] to satisfy the ONEX Any-type ban.
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.manifest_schema import CatalogManifest
from omnibase_infra.docker.catalog.resolver import ResolvedStack

# Env var name patterns that carry database connection strings.
_DB_URL_PATTERN = re.compile(r"(_DB_URL|_DSN|_DATABASE_URL)$")

# Host port patterns that indicate a host-side address was used instead of
# the Docker-internal address.  Matches ``localhost:<port>`` where the port
# is one of the well-known *external* (host-mapped) ports.
_LOCALHOST_HOST_PORT = re.compile(r"localhost:\d+")


def _validate_no_localhost_db_urls(manifest: CatalogManifest) -> None:
    """Reject manifests that pass DB URLs via required_env (host passthrough).

    Database connection strings (``*_DB_URL``, ``*_DSN``, ``*_DATABASE_URL``)
    must be hardcoded with Docker-internal addresses (e.g. ``postgres:5432``)
    rather than passed through from ``~/.omnibase/.env`` which uses
    ``localhost:<host-port>``.  This prevents the recurring class of bug
    where containers try to connect to ``localhost`` (i.e. themselves)
    instead of the Postgres container.

    Raises ``ValueError`` with a clear remediation message.
    """
    for var in manifest.required_env:
        if _DB_URL_PATTERN.search(var):
            raise ValueError(
                f"Service '{manifest.name}' declares '{var}' in required_env. "
                f"Database URLs must NOT be passed through from ~/.omnibase/.env "
                f"because that file uses localhost:<host-port> which is unreachable "
                f"from inside Docker containers. Move '{var}' to hardcoded_env with "
                f"Docker-internal addresses (e.g. postgres:5432)."
            )

    # Also check hardcoded_env and operational_defaults for localhost in DB URLs
    for section_name, section in [
        ("hardcoded_env", manifest.hardcoded_env),
        ("operational_defaults", manifest.operational_defaults),
    ]:
        for var, value in section.items():
            if _DB_URL_PATTERN.search(var) and _LOCALHOST_HOST_PORT.search(value):
                raise ValueError(
                    f"Service '{manifest.name}' has '{var}' in {section_name} "
                    f"containing 'localhost'. Database URLs inside Docker containers "
                    f"must use Docker-internal hostnames (e.g. postgres:5432), not "
                    f"localhost:<host-port>. Current value: {value}"
                )


def generate_compose(resolved: ResolvedStack) -> dict[str, object]:
    """Generate a docker-compose dict from a resolved stack."""
    services: dict[str, dict[str, object]] = {}
    all_volumes: set[str] = set()

    for name, manifest in resolved.manifests.items():
        # Validate DB URL configuration before generating compose output
        _validate_no_localhost_db_urls(manifest)
        svc: dict[str, object] = {}

        # Image
        svc["image"] = manifest.image

        # Container name
        if manifest.container_name:
            svc["container_name"] = manifest.container_name

        # Command
        if manifest.command:
            svc["command"] = manifest.command

        # Environment
        env: dict[str, str] = {}
        env.update(manifest.hardcoded_env)
        env.update(manifest.operational_defaults)
        env.update(manifest.catalog_env)

        # Add required env as ${VAR:?message} references
        for var in manifest.required_env:
            env[var] = f"${{{var}:?{var} must be set in ~/.omnibase/.env}}"

        # Inject bundle env only for runtime-layer entries
        if manifest.layer == EnumInfraLayer.RUNTIME:
            env.update(resolved.injected_env)

        if env:
            svc["environment"] = env

        # Ports
        if manifest.ports:
            svc["ports"] = [f"{manifest.ports.external}:{manifest.ports.internal}"]

        # Volumes
        if manifest.volumes:
            svc["volumes"] = list(manifest.volumes)
            for v in manifest.volumes:
                # Extract named volume (before :)
                vol_name = v.split(":")[0]
                if not vol_name.startswith(".") and not vol_name.startswith("/"):
                    all_volumes.add(vol_name)

        # Networks
        svc["networks"] = ["omnibase-infra-network"]

        # Healthcheck
        if manifest.healthcheck:
            svc["healthcheck"] = {
                "test": ["CMD-SHELL", manifest.healthcheck.test],
                "interval": f"{manifest.healthcheck.interval_s}s",
                "timeout": f"{manifest.healthcheck.timeout_s}s",
                "retries": manifest.healthcheck.retries,
                "start_period": f"{manifest.healthcheck.start_period_s}s",
            }

        # Restart
        svc["restart"] = manifest.restart

        # Stop grace period
        if manifest.stop_grace_period:
            svc["stop_grace_period"] = manifest.stop_grace_period

        # Deploy / Resources
        if manifest.resources:
            svc["deploy"] = {
                "resources": {
                    "limits": {
                        "cpus": manifest.resources.cpus,
                        "memory": manifest.resources.memory,
                    },
                    "reservations": {
                        "cpus": manifest.resources.cpus_reservation,
                        "memory": manifest.resources.memory_reservation,
                    },
                }
            }

        # Labels
        if manifest.labels:
            svc["labels"] = manifest.labels

        # Depends on
        if manifest.depends_on:
            deps: dict[str, dict[str, str]] = {}
            for dep in manifest.depends_on:
                deps[dep.service] = {"condition": dep.condition.value}
            svc["depends_on"] = deps

        services[name] = svc

    # Build top-level compose dict
    compose: dict[str, object] = {
        "name": "omnibase-infra",
        "services": services,
        "networks": {
            "omnibase-infra-network": {
                "name": "omnibase-infra-network",
                "driver": "bridge",
            }
        },
    }

    # Volumes
    if all_volumes:
        compose["volumes"] = {v: {"name": v} for v in sorted(all_volumes)}

    return compose
