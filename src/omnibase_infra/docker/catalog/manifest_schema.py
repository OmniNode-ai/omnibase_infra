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


@dataclass(frozen=True)
class PortMapping:
    """Explicit external/internal port pair."""

    external: int
    internal: int


@dataclass(frozen=True)
class HealthCheck:
    """Healthcheck with full timing parameters (matches compose healthcheck)."""

    test: str
    interval_s: int = 30
    timeout_s: int = 10
    retries: int = 3
    start_period_s: int = 10


@dataclass(frozen=True)
class DependsOnEntry:
    """A dependency on another catalog entry with an explicit condition."""

    service: str
    condition: EnumDependsOnCondition = EnumDependsOnCondition.SERVICE_STARTED


@dataclass(frozen=True)
class ResourceLimits:
    """Container resource constraints."""

    cpus: str = "1.0"
    memory: str = "768M"
    cpus_reservation: str = "0.25"
    memory_reservation: str = "128M"


@dataclass(frozen=True)
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
    container_name: str | None = None
    command: str | list[str] | None = None
    restart: str = "unless-stopped"
    labels: dict[str, str] = field(default_factory=dict)
    resources: ResourceLimits | None = None
    stop_grace_period: str | None = None
    # Per-entry env overrides (e.g., per-entry OTEL name)
    catalog_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.layer, str):
            object.__setattr__(self, "layer", EnumInfraLayer(self.layer))


@dataclass
class Bundle:
    """A named group of catalog entries that are deployed together."""

    name: str
    description: str
    services: list[str]  # entry names
    includes: list[str] = field(default_factory=list)
    inject_env: dict[str, str] = field(default_factory=dict)
    inject_required_env: list[str] = field(default_factory=list)

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
