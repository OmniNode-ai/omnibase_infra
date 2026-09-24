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
    ULimit,
)
from omnibase_infra.docker.catalog.model_optional_directory_bind_mount import (
    ModelOptionalDirectoryBindMount,
)

#: The compose project every bundle rendered under before OMN-19496. A bundle
#: that declares no ``project`` still renders here, with every historical name.
DEFAULT_PROJECT = "omnibase-infra"


@dataclass  # internal-dataclass-ok: docker-catalog-internal
class ResolvedStack:
    """Result of resolving bundles into a concrete set of catalog entries."""

    manifests: dict[str, CatalogManifest]
    required_env: set[str]
    injected_env: dict[str, str]
    injected_volumes: list[str] = field(default_factory=list)
    project: str = DEFAULT_PROJECT

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

    ulimits: dict[str, ULimit] = {}
    for name, limit in (raw.get("ulimits") or {}).items():
        ulimits[str(name)] = ULimit(
            soft=int(limit["soft"]),
            hard=int(limit["hard"]),
        )

    optional_directory_bind_mounts: list[ModelOptionalDirectoryBindMount] = []
    for mount in raw.get("optional_directory_bind_mounts", []):
        if not isinstance(mount, dict):
            raise ValueError("optional_directory_bind_mounts entries must be mappings")
        source_env = mount.get("source_env")
        container_path = mount.get("container_path")
        read_only = mount.get("read_only", True)
        if not isinstance(source_env, str) or not isinstance(container_path, str):
            raise ValueError(
                "optional directory bind mount source_env and container_path must be strings"
            )
        if not isinstance(read_only, bool):
            raise ValueError(
                "optional directory bind mount read_only must be a boolean"
            )
        optional_directory_bind_mounts.append(
            ModelOptionalDirectoryBindMount(
                source_env=source_env,
                container_path=container_path,
                read_only=read_only,
            )
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
        optional_directory_bind_mounts=optional_directory_bind_mounts,
        tmpfs=raw.get("tmpfs", []),
        depends_on=depends_on,
        container_name=raw.get("container_name"),
        command=raw.get("command"),
        restart=raw.get("restart", "unless-stopped"),
        labels=raw.get("labels", {}),
        resources=resources,
        ulimits=ulimits,
        stop_grace_period=raw.get("stop_grace_period"),
        catalog_env=raw.get("catalog_env", {}),
        extra_networks=raw.get("extra_networks", []),
    )


@dataclass  # internal-dataclass-ok: docker-catalog-internal
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
                    inject_volumes=bdef.get("inject_volumes", []),
                    project=bdef.get("project"),
                )

    def resolve(self, bundles: list[str]) -> ResolvedStack:
        """Resolve selected bundles into a concrete stack."""
        # Collect all bundle names preserving insertion order for deterministic
        # compose output (OMN-9345). dict[str, None] deduplicates while keeping order.
        all_bundle_names: dict[str, None] = {}
        for bundle_name in bundles:
            all_bundle_names[bundle_name] = None
            if bundle_name in self._bundles:
                included = self._bundles[bundle_name].resolve_includes(self._bundles)
                for inc in included:
                    all_bundle_names[inc] = None

        # Collect all entries from selected bundles
        selected_entries: dict[str, CatalogManifest] = {}
        required_env: set[str] = set()
        injected_env: dict[str, str] = {}
        injected_volumes: list[str] = []
        bundle_required_env: set[str] = set()
        project: str | None = None
        project_owner = ""

        for bundle_name in all_bundle_names:
            if bundle_name not in self._bundles:
                raise ValueError(
                    f"Unknown bundle '{bundle_name}'. "
                    f"Valid bundles: {sorted(self._bundles)}"
                )
            bundle = self._bundles[bundle_name]

            # Add entries
            for svc_name in bundle.services:
                if svc_name not in self._manifests:
                    raise ValueError(
                        f"Unknown service '{svc_name}' referenced by bundle "
                        f"'{bundle_name}'. Available services: "
                        f"{sorted(self._manifests)}"
                    )
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
            bundle_required_env.update(bundle.inject_required_env)

            for volume in bundle.inject_volumes:
                if volume not in injected_volumes:
                    injected_volumes.append(volume)

            if bundle.project is not None:
                if project is not None and project != bundle.project:
                    raise ValueError(
                        f"Compose project conflict: bundle '{project_owner}' "
                        f"renders under '{project}' and bundle '{bundle_name}' "
                        f"under '{bundle.project}'. One stack is one project."
                    )
                project = bundle.project
                project_owner = bundle_name

        # Transitively resolve service dependencies (BFS until no new deps found)
        pending: list[CatalogManifest] = list(selected_entries.values())
        while pending:
            next_pending: list[CatalogManifest] = []
            for manifest in pending:
                for dep in manifest.depends_on:
                    if dep.service not in self._manifests:
                        raise ValueError(
                            f"Unknown dependency '{dep.service}' referenced by "
                            f"service '{manifest.name}'. Available services: "
                            f"{sorted(self._manifests)}"
                        )
                    if dep.service not in selected_entries:
                        dep_manifest = self._manifests[dep.service]
                        selected_entries[dep.service] = dep_manifest
                        required_env.update(dep_manifest.required_env)
                        next_pending.append(dep_manifest)
            pending = next_pending

        # OMN-19496: a bundle's ``inject_env`` value is what the generator
        # renders for that var on every runtime-layer entry, over the entry's
        # own ``${VAR:?}`` reference. Such a var is therefore satisfied by the
        # bundle, not by the operator, when every selected entry that declares
        # it is runtime-layer. An infrastructure entry never receives injected
        # env, so a var it requires stays required, and a var a bundle lists in
        # ``inject_required_env`` stays required by construction.
        for var in list(required_env):
            if var not in injected_env or var in bundle_required_env:
                continue
            requirers = [m for m in selected_entries.values() if var in m.required_env]
            if requirers and all(m.layer == EnumInfraLayer.RUNTIME for m in requirers):
                required_env.discard(var)

        return ResolvedStack(
            manifests=selected_entries,
            required_env=required_env,
            injected_env=injected_env,
            injected_volumes=injected_volumes,
            project=project or DEFAULT_PROJECT,
        )
