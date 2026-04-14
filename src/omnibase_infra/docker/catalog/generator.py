# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose generator — renders a ResolvedStack into docker-compose YAML."""

from __future__ import annotations

# Compose YAML values are heterogeneous dicts. We use dict[str, object]
# instead of dict[str, Any] to satisfy the ONEX Any-type ban.
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.resolver import ResolvedStack


def generate_compose(resolved: ResolvedStack) -> dict[str, object]:
    """Generate a docker-compose dict from a resolved stack."""
    services: dict[str, dict[str, object]] = {}
    all_volumes: set[str] = set()
    all_extra_networks: set[str] = set()

    for name, manifest in resolved.manifests.items():
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
        networks: list[str] = ["omnibase-infra-network"]
        if manifest.extra_networks:
            networks.extend(manifest.extra_networks)
            all_extra_networks.update(manifest.extra_networks)
        svc["networks"] = networks

        # Healthcheck
        if manifest.healthcheck:
            if isinstance(manifest.healthcheck.test, list):
                test_cmd: list[str] = ["CMD", *manifest.healthcheck.test]
            else:
                test_cmd = ["CMD-SHELL", manifest.healthcheck.test]
            svc["healthcheck"] = {
                "test": test_cmd,
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

    # Build top-level networks block
    top_networks: dict[str, object] = {
        "omnibase-infra-network": {
            "name": "omnibase-infra-network",
            "driver": "bridge",
        }
    }
    for net in sorted(all_extra_networks):
        if net == "omnibase-infra-network":
            continue  # never overwrite the default bridge network as external
        top_networks[net] = {"name": net, "external": True}

    # Build top-level compose dict
    compose: dict[str, object] = {
        "name": "omnibase-infra",
        "services": services,
        "networks": top_networks,
    }

    # Volumes
    if all_volumes:
        compose["volumes"] = {v: {"name": v} for v in sorted(all_volumes)}

    return compose
