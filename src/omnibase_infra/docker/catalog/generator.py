# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose generator — renders a ResolvedStack into docker-compose YAML."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

# Compose YAML values are heterogeneous dicts. We use dict[str, object]
# instead of dict[str, Any] to satisfy the ONEX Any-type ban.
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.resolver import DEFAULT_PROJECT, ResolvedStack

_RUNTIME_IMAGE_BUILD_SERVICE = "omninode-runtime"
_RUNTIME_IMAGE = "runtime:latest"
_DEFAULT_NETWORK = "omnibase-infra-network"


def _scoped(project: str, name: str) -> str:
    """Project-scope one Docker object name (OMN-19496).

    The default project keeps every historical name byte-for-byte, so no
    existing render changes. Any other project prefixes the name, which is what
    lets a laptop profile run beside another compose project on one Docker host
    without sharing a container, network, volume or image tag with it.
    """
    if project == DEFAULT_PROJECT:
        return name
    return f"{project}-{name}"


# OMN-18478. The PostToolUse secret redactor rewrites Bash tool OUTPUT, replacing
# a credential with this token. It does not touch Write/Edit content, so the
# token only ever reaches a file when an author used already-redacted tool output
# as the source text for an edit -- which is how
# docker/catalog/services/tenant-projection-writer.yaml came to bind it as a
# postgres password. Rendering it produces a container whose DSN cannot
# authenticate, and the failure surfaces at first query rather than at render.
#
# Built from parts rather than spelled: the literal in a source file is itself
# rewritten in any Bash preview of this module, so a reader greps it and sees the
# token instead of the code. The construction stays legible under redaction.
_REDACTION_PLACEHOLDER = "*" * 3 + "REDACTED" + "*" * 3


def _reject_redaction_placeholder(service: str, env: Mapping[str, str]) -> None:
    """Refuse an environment binding carrying the redactor's replacement token.

    Raised rather than logged: a rendered placeholder is indistinguishable from a
    real value to every downstream consumer of the compose file, so the render
    must not succeed.
    """
    offenders = sorted(k for k, v in env.items() if _REDACTION_PLACEHOLDER in v)
    if offenders:
        raise ValueError(
            f"{service}: environment {offenders} carries the secret redactor's "
            f"replacement token instead of a value. This is redacted tool output "
            f"that was written into a catalog manifest; recover the real value "
            f"or bind it as a ${{VAR:?...}} expansion."
        )


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


def _runtime_image_build(project: str = DEFAULT_PROJECT) -> dict[str, object]:
    """Return the canonical build stanza for the shared runtime image.

    ``OMNI_HOME`` is deliberately absent (OMN-16852). It is an internal build
    input that only a ``BUILD_SOURCE=workspace`` build reads, and every
    sanctioned workspace build supplies it as ``--build-arg`` after refusing an
    unset value. Rendering ``${OMNI_HOME:-}`` here only handed the build a
    silent empty default; without it, the Dockerfile's workspace guard fails
    fast when the value is missing.
    """
    return {
        "context": "..",
        "dockerfile": "docker/Dockerfile.runtime",
        "args": {
            "BUILD_SOURCE": "${BUILD_SOURCE:-release}",
            "EXPECTED_BUILD_SOURCE": "${EXPECTED_BUILD_SOURCE:-release}",
            "RUNTIME_VERSION": "${RUNTIME_VERSION:-0.1.0}",
            "BUILD_DATE": "${BUILD_DATE:-}",
            "VCS_REF": "${VCS_REF:-}",
            "GIT_SHA": "${GIT_SHA:-unknown}",
            "RUNTIME_SOURCE_HASH": "${RUNTIME_SOURCE_HASH:-unknown}",
            "COMPOSE_PROJECT": f"${{COMPOSE_PROJECT:-{project}}}",
        },
    }


def generate_compose(
    resolved: ResolvedStack, *, environment: Mapping[str, str] | None = None
) -> dict[str, object]:
    """Generate a docker-compose dict from a resolved stack."""
    configured_environment = environment or {}
    project = resolved.project
    services: dict[str, dict[str, object]] = {}
    all_volumes: set[str] = set()
    all_extra_networks: set[str] = set()

    for name, manifest in resolved.manifests.items():
        svc: dict[str, object] = {}

        # Image. The locally built runtime image is project-scoped so a second
        # project's build never retags the image another project runs.
        if manifest.image == _RUNTIME_IMAGE:
            svc["image"] = _scoped(project, manifest.image)
        else:
            svc["image"] = manifest.image
        if name == _RUNTIME_IMAGE_BUILD_SERVICE and manifest.image == _RUNTIME_IMAGE:
            svc["build"] = _runtime_image_build(project)

        # Container name. A non-default project names every container after
        # the project and the service, never after the historical name.
        if project != DEFAULT_PROJECT:
            svc["container_name"] = _scoped(project, name)
        elif manifest.container_name:
            svc["container_name"] = manifest.container_name

        # Command
        if manifest.command:
            svc["command"] = manifest.command

        # Environment
        env: dict[str, str] = {}
        env.update(manifest.hardcoded_env)
        env.update(manifest.operational_defaults)
        env.update(manifest.catalog_env)
        _reject_redaction_placeholder(name, env)

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
        if manifest.layer == EnumInfraLayer.RUNTIME:
            volumes.extend(resolved.injected_volumes)
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
        networks: list[str] = [_DEFAULT_NETWORK]
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
        _DEFAULT_NETWORK: {
            "name": _scoped(project, "network")
            if project != DEFAULT_PROJECT
            else _DEFAULT_NETWORK,
            "driver": "bridge",
        }
    }
    for net in sorted(all_extra_networks):
        if net == _DEFAULT_NETWORK:
            continue  # never overwrite the default bridge network as external
        top_networks[net] = {"name": net, "external": True}

    # Build top-level compose dict
    compose: dict[str, object] = {
        "name": project,
        "services": services,
        "networks": top_networks,
    }

    # Volumes. Declared names are global to the Docker host, so a non-default
    # project scopes them; the keys services reference stay unchanged.
    if all_volumes:
        compose["volumes"] = {
            v: {"name": _scoped(project, v)} for v in sorted(all_volumes)
        }

    return compose
