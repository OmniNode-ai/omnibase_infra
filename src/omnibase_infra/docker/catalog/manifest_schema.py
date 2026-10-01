# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Catalog manifest and bundle schema for the ONEX infrastructure catalog.

The schema models in this module are the authoritative definition for
catalog manifest YAML files. Every field present in a manifest YAML
must have a corresponding field here. Conversely, every field here
must be populated when loading a manifest YAML. No loose dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.docker.catalog.enum_depends_on_condition import (
    EnumDependsOnCondition,
)
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.model_optional_directory_bind_mount import (
    ModelOptionalDirectoryBindMount,
)


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class PortMapping:
    """Explicit external/internal port pair."""

    external: int
    internal: int


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class HealthCheck:
    """Healthcheck with full timing parameters (matches compose healthcheck).

    ``test`` accepts two forms:
    - **str** — rendered as ``["CMD-SHELL", test]`` (requires ``/bin/sh``).
    - **list[str]** — rendered as ``["CMD", *test]`` for distroless images
      that lack a shell (e.g. Phoenix).
    """

    test: str | list[str]
    interval_s: int = 30
    timeout_s: int = 10
    retries: int = 3
    start_period_s: int = 10


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class DependsOnEntry:
    """A dependency on another catalog entry with an explicit condition."""

    service: str
    condition: EnumDependsOnCondition = EnumDependsOnCondition.SERVICE_STARTED


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class ResourceLimits:
    """Container resource constraints."""

    cpus: str = "1.0"
    memory: str = "768M"
    cpus_reservation: str = "0.25"
    memory_reservation: str = "128M"


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class ULimit:
    """Container ulimit soft/hard pair for docker compose."""

    soft: int
    hard: int


@dataclass(frozen=True)  # internal-dataclass-ok: docker-catalog-internal
class CatalogManifest:
    """Declaration of a single deployable catalog entry.

    Every field corresponds to a key in the manifest YAML.
    No freeform dicts -- all structure is typed.
    """

    name: str
    description: str
    image: str
    layer: EnumInfraLayer
    required_env: list[str]
    hardcoded_env: dict[str, str]
    operational_defaults: dict[str, str]
    ports: PortMapping | None
    healthcheck: HealthCheck | None
    volumes: list[str]
    depends_on: list[DependsOnEntry]
    # Optional fields with sane defaults
    optional_directory_bind_mounts: list[ModelOptionalDirectoryBindMount] = field(
        default_factory=list
    )
    tmpfs: list[str] = field(default_factory=list)
    container_name: str | None = None
    command: str | list[str] | None = None
    # OMN-19496: overrides the image ENTRYPOINT. Without it a one-shot that runs
    # a shell script on an image whose entrypoint wraps a CLI (redpanda's
    # /entrypoint.sh wraps rpk) passes the script to that CLI as arguments.
    entrypoint: list[str] | None = None
    restart: str = "unless-stopped"
    labels: dict[str, str] = field(default_factory=dict)
    resources: ResourceLimits | None = None
    ulimits: dict[str, ULimit] = field(default_factory=dict)
    stop_grace_period: str | None = None
    # Per-entry env overrides (e.g., per-entry OTEL name)
    catalog_env: dict[str, str] = field(default_factory=dict)
    # Additional networks beyond the default omnibase-infra-network.
    # Each entry is a network name; set external=True in the top-level compose
    # networks block. Use when a service must reach containers on a separate
    # compose-managed network (e.g. omnimemory-network for Memgraph).
    extra_networks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.layer, str):
            object.__setattr__(self, "layer", EnumInfraLayer(self.layer))


@dataclass  # internal-dataclass-ok: docker-catalog-internal
class Bundle:
    """A named group of catalog entries that are deployed together."""

    name: str
    description: str
    services: list[str]  # entry names
    includes: list[str] = field(default_factory=list)
    inject_env: dict[str, str] = field(default_factory=dict)
    inject_required_env: list[str] = field(default_factory=list)
    # OMN-19496: bind mounts added to every runtime-layer entry of the resolved
    # stack, on the same rule as ``inject_env``. A laptop profile uses this to
    # mount its own Bifrost lane overlay without editing shared manifests.
    inject_volumes: list[str] = field(default_factory=list)
    # OMN-19496: the compose project the bundle renders under. ``None`` keeps
    # the historical ``omnibase-infra`` project and every historical name. A
    # bundle that names its own project gets project-scoped container,
    # network, volume and runtime-image names, so it can run beside any other
    # compose project on the same Docker host.
    project: str | None = None
    # OMN-19972: the host address every published port binds to. ``None``
    # keeps the historical ``<external>:<internal>`` form, which binds all
    # interfaces; the lab lanes render from the same shared manifests and are
    # meant to be reachable. The laptop profile sets ``127.0.0.1`` so a
    # developer's database and broker are not published to their network.
    publish_host: str | None = None
    # OMN-19972: replacement defaults for ``${VAR:-default}`` references in
    # every entry's command and environment. A shared manifest's default is
    # right for the lab lanes that render it; a bundle that must not carry it
    # (the laptop profile and the lab's LAN address) overrides it here instead
    # of editing the shared manifest.
    env_default_overrides: dict[str, str] = field(default_factory=dict)
    # OMN-19972: a replacement external (host) port per entry name. The shared
    # manifest's port is right for the lab lanes that render it; the laptop
    # profile must not take the lab's projection API port, so it overrides the
    # host side here and leaves the container port alone.
    port_overrides: dict[str, int] = field(default_factory=dict)
    # OMN-19972: extra start-order dependencies per entry name, as
    # ``{entry: {dependency: condition}}``. On the laptop the projection API
    # must wait for the kernel that provisions its exposure topics; other
    # bundles that render the projection API may not run that kernel at all,
    # so the dependency belongs to the bundle, not to the shared manifest.
    extra_depends_on: dict[str, dict[str, str]] = field(default_factory=dict)

    def resolve_includes(
        self,
        bundles: dict[str, Bundle],
        visited: set[str] | None = None,
    ) -> list[str]:
        """Return flat list of all included bundle names, detecting circular deps."""
        if visited is None:
            visited = set()
        if self.name in visited:
            raise ValueError(f"circular dependency detected: {self.name} -> {visited}")
        visited.add(self.name)
        result: list[str] = []
        for inc in self.includes:
            result.append(inc)
            if inc in bundles:
                result.extend(bundles[inc].resolve_includes(bundles, visited.copy()))
        return result

    def all_required_env(self, catalog: dict[str, CatalogManifest]) -> set[str]:
        """Collect all required env vars from this bundle's entries and dependencies."""
        env: set[str] = set()
        env.update(self.inject_required_env)
        for svc_name in self.services:
            if svc_name in catalog:
                svc = catalog[svc_name]
                env.update(svc.required_env)
                for dep in svc.depends_on:
                    if dep.service in catalog:
                        env.update(catalog[dep.service].required_env)
        return env


# Backwards-compatible aliases for plan-specified names
ServiceManifest = CatalogManifest
ServiceLayer = EnumInfraLayer
DependsOnCondition = EnumDependsOnCondition
