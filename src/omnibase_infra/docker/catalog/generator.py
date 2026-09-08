# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose generator — renders a ResolvedStack into docker-compose YAML."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

# Compose YAML values are heterogeneous dicts. We use dict[str, object]
# instead of dict[str, Any] to satisfy the ONEX Any-type ban.
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.resolver import ResolvedStack

_RUNTIME_IMAGE_BUILD_SERVICE = "omninode-runtime"


# Compose default for an optional directory bind mount whose source variable is
# unset. It must be a DIRECTORY: a file fallback such as ``/dev/null`` is
# invalid for a directory target, which is the OMN-13248 defect that made the
# mount conditional in the first place. Docker creates a missing bind source
# directory, so this resolves to an empty, read-only directory on any host --
# the same outcome the container saw when the mount was omitted, and the
# coding-agent handler still fails closed on absent credentials.
_ABSENT_OPTIONAL_BIND_SOURCE = "/var/lib/omninode/optional-bind-source-absent"


def _render_optional_directory_bind_mount(
    *,
    source_env: str,
    container_path: str,
    read_only: bool,
    environment: Mapping[str, str],
) -> str:
    """Render an optional directory mount from the catalog declaration alone.

    The rendered entry is a function of the catalog declaration and nothing
    else. It used to be a function of the RENDER HOST as well (OMN-17291): the
    mount was emitted only when ``source_env`` was set in the render process's
    environment AND pointed at an existing directory, and dropped otherwise. Two
    hosts therefore rendered two different compose files from one commit -- and
    because the emitted expression was a compose REQUIRED-var (``${VAR:?}``),
    they rendered two different required-var NAME sets. The committed
    declaration in ``docker/generated-compose-required-env.manifest.txt`` cannot
    be exact for both, so the parity test asserting it passed on a workstation
    carrying ambient coding-agent credentials and failed on a lab host carrying
    none, for the same tree.

    The source is now always emitted with a compose DEFAULT rather than a
    required-var expression, so the mount introduces no required-var name at
    all and every render interpolates on a host that supplies no value --
    which ``docker compose config`` on the generated render does, on the
    stability lane and in the deploy agent's own compose_gen path.

    A configured source is still validated as an existing absolute directory at
    render time, so an explicitly set but unusable value fails closed here
    rather than at container start.
    """
    raw_source = environment.get(source_env)
    if raw_source is not None and raw_source.strip():
        if raw_source != raw_source.strip():
            raise ValueError(f"{source_env} must not contain surrounding whitespace")
        source_path = Path(raw_source)
        if not source_path.is_absolute() or not source_path.is_dir():
            raise ValueError(
                f"{source_env} must point to an existing absolute directory"
            )

    mode = ":ro" if read_only else ""
    return f"${{{source_env}:-{_ABSENT_OPTIONAL_BIND_SOURCE}}}:{container_path}{mode}"


def _runtime_image_build() -> dict[str, object]:
    """Return the canonical build stanza for the shared runtime image."""
    return {
        "context": "..",
        "dockerfile": "docker/Dockerfile.runtime",
        "args": {
            "BUILD_SOURCE": "${BUILD_SOURCE:-release}",
            "EXPECTED_BUILD_SOURCE": "${EXPECTED_BUILD_SOURCE:-release}",
            "OMNI_HOME": "${OMNI_HOME:-}",
            "RUNTIME_VERSION": "${RUNTIME_VERSION:-0.1.0}",
            "BUILD_DATE": "${BUILD_DATE:-}",
            "VCS_REF": "${VCS_REF:-}",
            "GIT_SHA": "${GIT_SHA:-unknown}",
            "RUNTIME_SOURCE_HASH": "${RUNTIME_SOURCE_HASH:-unknown}",
            "COMPOSE_PROJECT": "${COMPOSE_PROJECT:-omnibase-infra}",
        },
    }


def generate_compose(
    resolved: ResolvedStack, *, environment: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Generate a docker-compose dict from a resolved stack."""
    configured_environment = environment or {}
    services: dict[str, dict[str, object]] = {}
    all_volumes: set[str] = set()
    all_extra_networks: set[str] = set()

    for name, manifest in resolved.manifests.items():
        svc: dict[str, object] = {}

        # Image
        svc["image"] = manifest.image
        if name == _RUNTIME_IMAGE_BUILD_SERVICE and manifest.image == "runtime:latest":
            svc["build"] = _runtime_image_build()

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
        volumes = list(manifest.volumes)
        for mount in manifest.optional_directory_bind_mounts:
            volumes.append(
                _render_optional_directory_bind_mount(
                    source_env=mount.source_env,
                    container_path=mount.container_path,
                    read_only=mount.read_only,
                    environment=configured_environment,
                )
            )

        if volumes:
            svc["volumes"] = volumes
            for v in volumes:
                # Extract named volume (before :)
                vol_name = v.split(":")[0]
                # Skip bind-mount sources: relative paths (`.`), absolute paths
                # (`/`), and operator-supplied host paths expressed as a shell
                # variable expansion (`${VAR:-/default}`). Only a bare token is a
                # Docker named volume that must be declared in the top-level
                # `volumes:` block. (OMN-13248: the coding-agent cred bind-mounts
                # use `${CODING_AGENT_*_CREDS_HOST_DIR:-/dev/null}` sources.)
                if (
                    not vol_name.startswith(".")
                    and not vol_name.startswith("/")
                    and not vol_name.startswith("${")
                ):
                    all_volumes.add(vol_name)

        # Tmpfs mounts
        if manifest.tmpfs:
            svc["tmpfs"] = list(manifest.tmpfs)

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

        # Ulimits
        if manifest.ulimits:
            svc["ulimits"] = {
                name: {"soft": limit.soft, "hard": limit.hard}
                for name, limit in sorted(manifest.ulimits.items())
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
