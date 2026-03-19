# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Catalog resolver — loads manifests and bundles, resolves transitive deps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from omnibase_infra.docker.catalog.enum_depends_on_condition import (
    EnumDependsOnCondition,
)
from omnibase_infra.docker.catalog.enum_infra_layer import EnumInfraLayer
from omnibase_infra.docker.catalog.manifest_schema import (
    Bundle,
    CatalogManifest,
    DependsOnEntry,
    HealthCheck,
    PortMapping,
    ResourceLimits,
)


@dataclass
class ResolvedStack:
    """Result of resolving bundles into a concrete set of catalog entries."""

    manifests: dict[str, CatalogManifest]
    required_env: set[str]
    injected_env: dict[str, str]

    @property
    def service_names(self) -> set[str]:
        return set(self.manifests.keys())


def _load_manifest(path: Path) -> CatalogManifest:
    """Load a single manifest YAML into a CatalogManifest."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    ports = None
    if raw.get("ports"):
        ports = PortMapping(
            external=raw["ports"]["external"], internal=raw["ports"]["internal"]
        )

    healthcheck = None
    if raw.get("healthcheck"):
        hc = raw["healthcheck"]
        healthcheck = HealthCheck(
            test=hc["test"],
            interval_s=hc.get("interval_s", 30),
            timeout_s=hc.get("timeout_s", 10),
            retries=hc.get("retries", 3),
            start_period_s=hc.get("start_period_s", 10),
        )

    depends_on = []
    for dep in raw.get("depends_on", []):
        if isinstance(dep, dict):
            depends_on.append(
                DependsOnEntry(
                    service=dep["service"],
                    condition=EnumDependsOnCondition(
                        dep.get("condition", "service_started")
                    ),
                )
            )
        else:
            depends_on.append(DependsOnEntry(service=str(dep)))

    resources = None
    if raw.get("resources"):
        r = raw["resources"]
        resources = ResourceLimits(
            cpus=str(r.get("cpus", "1.0")),
            memory=str(r.get("memory", "768M")),
            cpus_reservation=str(r.get("cpus_reservation", "0.25")),
            memory_reservation=str(r.get("memory_reservation", "128M")),
        )

    return CatalogManifest(
        name=raw["name"],
        description=raw.get("description", ""),
        image=raw["image"],
        layer=EnumInfraLayer(raw["layer"]),
        required_env=raw.get("required_env", []),
        hardcoded_env=raw.get("hardcoded_env", {}),
        operational_defaults=raw.get("operational_defaults", {}),
        ports=ports,
        healthcheck=healthcheck,
        volumes=raw.get("volumes", []),
        depends_on=depends_on,
        container_name=raw.get("container_name"),
        command=raw.get("command"),
        restart=raw.get("restart", "unless-stopped"),
        labels=raw.get("labels", {}),
        resources=resources,
        stop_grace_period=raw.get("stop_grace_period"),
        catalog_env=raw.get("catalog_env", {}),
    )


@dataclass
class CatalogResolver:
    """Loads catalog manifests and bundles, resolves selected bundles."""

    catalog_dir: str
    _manifests: dict[str, CatalogManifest] = field(default_factory=dict, init=False)
    _bundles: dict[str, Bundle] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        catalog_path = Path(self.catalog_dir)
        # Load all manifests
        services_dir = catalog_path / "services"
        if services_dir.exists():
            for manifest_file in services_dir.glob("*.yaml"):
                manifest = _load_manifest(manifest_file)
                self._manifests[manifest.name] = manifest

        # Load bundles
        bundles_file = catalog_path / "bundles.yaml"
        if bundles_file.exists():
            with open(bundles_file) as f:
                raw_bundles = yaml.safe_load(f) or {}
            for name, bdef in raw_bundles.items():
                self._bundles[name] = Bundle(
                    name=name,
                    description=bdef.get("description", ""),
                    services=bdef.get("services", []),
                    includes=bdef.get("includes", []),
                    inject_env=bdef.get("inject_env", {}),
                    inject_required_env=bdef.get("inject_required_env", []),
                )

    def resolve(self, bundles: list[str]) -> ResolvedStack:
        """Resolve selected bundles into a concrete stack."""
        # Collect all bundle names (including transitive includes)
        all_bundle_names: set[str] = set()
        for bundle_name in bundles:
            all_bundle_names.add(bundle_name)
            if bundle_name in self._bundles:
                included = self._bundles[bundle_name].resolve_includes(self._bundles)
                all_bundle_names.update(included)

        # Collect all entries from selected bundles
        selected_entries: dict[str, CatalogManifest] = {}
        required_env: set[str] = set()
        injected_env: dict[str, str] = {}

        for bundle_name in all_bundle_names:
            if bundle_name not in self._bundles:
                raise ValueError(
                    f"Unknown bundle '{bundle_name}'. "
                    f"Valid bundles: {sorted(self._bundles)}"
                )
            bundle = self._bundles[bundle_name]

            # Add entries
            for svc_name in bundle.services:
                if svc_name in self._manifests:
                    manifest = self._manifests[svc_name]
                    selected_entries[svc_name] = manifest
                    required_env.update(manifest.required_env)

            # Add injected env
            for k, v in bundle.inject_env.items():
                if k in injected_env and injected_env[k] != v:
                    raise ValueError(
                        f"Env var conflict: {k} set to '{injected_env[k]}' by one "
                        f"bundle and '{v}' by bundle '{bundle_name}'"
                    )
                injected_env[k] = v

            # Add required env from bundle
            required_env.update(bundle.inject_required_env)

        # Transitively resolve service dependencies (BFS until no new deps found)
        pending: list[CatalogManifest] = list(selected_entries.values())
        while pending:
            next_pending: list[CatalogManifest] = []
            for manifest in pending:
                for dep in manifest.depends_on:
                    if (
                        dep.service in self._manifests
                        and dep.service not in selected_entries
                    ):
                        dep_manifest = self._manifests[dep.service]
                        selected_entries[dep.service] = dep_manifest
                        required_env.update(dep_manifest.required_env)
                        next_pending.append(dep_manifest)
            pending = next_pending

        return ResolvedStack(
            manifests=selected_entries,
            required_env=required_env,
            injected_env=injected_env,
        )
