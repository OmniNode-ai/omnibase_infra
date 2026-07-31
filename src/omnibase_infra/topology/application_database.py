# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Authoritative application-database topology loading and parity validation.

The checked-in ``instances/*.yaml`` files are the only platform authority used by
this module. Host-local ``~/.omnibase/topology.yaml`` is deliberately not searched:
local setup may render a projection there, but it cannot override platform truth.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.core import ModelDeploymentTopology
from omnibase_infra.docker.catalog.resolver import _load_manifest
from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from omnibase_infra.topology.models import (
    ModelApplicationDatabaseTopologyProfile,
    ModelApplicationDatabaseTopologyProfileCatalog,
    ModelDockerDatabaseConsumerCatalog,
)

APPLICATION_DATABASE_REF = "application"
APPLICATION_DATABASE_PHYSICAL_NAME = "omnidash_analytics"

_EXPECTED_PROFILE_INSTANCE_MAP = {
    "local": "local",
    "test": "local",
    "stability-test": "local",
    "judge": "local",
    "prod": "local",
    "onex-dev": "onex-dev",
    "onex-prod": "onex-prod",
}
_EXPECTED_PROFILE_INJECTION_SURFACES = {
    "local": ("OmniNode-ai/omnibase_infra", "docker/docker-compose.infra.yml"),
    "test": ("OmniNode-ai/omnibase_infra", "docker/docker-compose.e2e.yml"),
    "stability-test": (
        "OmniNode-ai/omnibase_infra",
        "docker/docker-compose.stability-test.yml",
    ),
    "judge": ("OmniNode-ai/omnibase_infra", "docker/docker-compose.judge.yml"),
    "prod": ("OmniNode-ai/omnibase_infra", "docker/docker-compose.prod.yml"),
    "onex-dev": (
        "OmniNode-ai/omninode_infra",
        "k8s/onex-dev/runtime/configmap.yaml",
    ),
    "onex-prod": (
        "OmniNode-ai/omninode_infra",
        "k8s/onex-prod/runtime/configmap.yaml",
    ),
}
_EXPECTED_RUNTIME_POLICY_PROFILE_MAP = {
    "dev": "local",
    "stability-test": "stability-test",
    "judge": "judge",
    "prod": "prod",
}
SUPPORTED_TOPOLOGY_PROFILES = frozenset(_EXPECTED_PROFILE_INSTANCE_MAP)
TOPOLOGY_PROFILE_INSTANCE_MAP = MappingProxyType(_EXPECTED_PROFILE_INSTANCE_MAP)
# Compatibility name for the draft API. New consumers must use the explicit
# profile terminology because ONEX_ENVIRONMENT is a separate event namespace.
SUPPORTED_ENVIRONMENTS = SUPPORTED_TOPOLOGY_PROFILES

_TOPOLOGY_INSTANCE_ROOT = Path(__file__).resolve().parent / "instances"
_PROFILE_CATALOG_PATH = (
    Path(__file__).resolve().parent / "application_database_profiles.yaml"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TOPOLOGY_REPOSITORY = "OmniNode-ai/omnibase_infra"
_TOPOLOGY_SOURCE_PREFIX = "src/omnibase_infra/topology/instances"
_PROFILE_CATALOG_SOURCE_PATH = (
    "src/omnibase_infra/topology/application_database_profiles.yaml"
)
_TOPOLOGY_PROFILE_ENV_VAR = "ONEX_DATABASE_TOPOLOGY_PROFILE"
_TOPOLOGY_PROFILE_LINE = re.compile(
    rf"^\s*{_TOPOLOGY_PROFILE_ENV_VAR}:\s*[\"']?(?P<profile>[a-z0-9-]+)"
    r"[\"']?\s*(?:#.*)?$",
    re.MULTILINE,
)

_EXPECTED_SCHEMAS = {
    "tenant": EnumDatabaseSchemaDomain.TENANT,
    "omninode_internal": EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,
    "platform_catalog": EnumDatabaseSchemaDomain.PLATFORM_CATALOG,
}
_EXPECTED_SCHEMA_OWNERS = {
    "tenant": "owner_onex_tenant",
    "omninode_internal": "owner_omninode_internal",
    "platform_catalog": "owner_platform_catalog",
}
_EXPECTED_BINDING_PRINCIPALS = {
    "onex_api": "onex_api",
    "tenant_projection": "tenant_projection_writer",
    "app_dashboard": "app_dashboard",
    "omninode_runtime_service": "omninode_runtime",
}
_EXPECTED_BINDING_DSN_ENVS = {
    "local": {
        "onex_api": "OMNINODE_CLOUD_DB_URL",
        "tenant_projection": "OMNIDASH_ANALYTICS_DB_URL",
        "app_dashboard": "OMNIDASH_ANALYTICS_DB_URL",
        "omninode_runtime_service": "OMNINODE_INTERNAL_DB_URL",
    },
    "onex-dev": {
        "onex_api": "OMNINODE_CLOUD_DB_URL",
        "tenant_projection": "OMNIDASH_ANALYTICS_DB_URL",
        "app_dashboard": "DATABASE_URL",
        "omninode_runtime_service": "OMNINODE_INTERNAL_DB_URL",
    },
    "onex-prod": {
        "onex_api": "OMNINODE_CLOUD_DB_URL",
        "tenant_projection": "OMNIDASH_ANALYTICS_DB_URL",
        "app_dashboard": "DATABASE_URL",
        "omninode_runtime_service": "OMNINODE_INTERNAL_DB_URL",
    },
}


def _load_profile_catalog(
    profile_catalog_path: Path | None = None,
) -> ModelApplicationDatabaseTopologyProfileCatalog:
    """Load and validate the exact checked-in profile-to-instance contract."""
    path = profile_catalog_path or _PROFILE_CATALOG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    catalog = ModelApplicationDatabaseTopologyProfileCatalog.model_validate(raw)
    if set(catalog.profiles) != SUPPORTED_TOPOLOGY_PROFILES:
        raise ValueError(
            "Database topology profile set drift: expected "
            f"{sorted(SUPPORTED_TOPOLOGY_PROFILES)}, got "
            f"{sorted(catalog.profiles)}"
        )
    actual_instance_map = {
        profile: binding.instance for profile, binding in catalog.profiles.items()
    }
    if actual_instance_map != _EXPECTED_PROFILE_INSTANCE_MAP:
        raise ValueError(
            "Database topology profile/instance drift: expected "
            f"{_EXPECTED_PROFILE_INSTANCE_MAP}, got {actual_instance_map}"
        )
    actual_surfaces = {
        profile: (binding.deployment_repository, binding.injection_path)
        for profile, binding in catalog.profiles.items()
    }
    if actual_surfaces != _EXPECTED_PROFILE_INJECTION_SURFACES:
        raise ValueError("Database topology profile injection surface drift")
    actual_runtime_profiles = {
        binding.runtime_policy_profile: profile
        for profile, binding in catalog.profiles.items()
        if binding.runtime_policy_profile is not None
    }
    if actual_runtime_profiles != _EXPECTED_RUNTIME_POLICY_PROFILE_MAP:
        raise ValueError("Database topology runtime-policy profile drift")
    return catalog


def _resolve_profile(
    profile: str,
    profile_catalog_path: Path | None = None,
) -> ModelApplicationDatabaseTopologyProfile:
    """Resolve one exact profile without environment inference or fallback."""
    catalog = _load_profile_catalog(profile_catalog_path)
    binding = catalog.profiles.get(profile)
    if binding is None:
        raise ValueError(
            f"Unsupported database topology profile '{profile}'; expected one of "
            f"{sorted(SUPPORTED_TOPOLOGY_PROFILES)}"
        )
    return binding


def _topology_instance_path(
    instance: str,
    topology_root: Path | None = None,
) -> Path:
    """Return one allowlisted checked-in topology instance without fallback."""
    if instance not in _EXPECTED_BINDING_DSN_ENVS:
        raise ValueError(f"Unsupported database topology instance '{instance}'")
    root = topology_root if topology_root is not None else _TOPOLOGY_INSTANCE_ROOT
    path = root / f"{instance}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required checked-in deployment topology does not exist: {path}"
        )
    return path


def load_topology_profile(
    profile: str,
    topology_root: Path | None = None,
    *,
    profile_catalog_path: Path | None = None,
) -> ModelDeploymentTopology:
    """Load a topology by its independent database-topology profile."""
    binding = _resolve_profile(profile, profile_catalog_path)
    topology = ModelDeploymentTopology.from_yaml(
        _topology_instance_path(binding.instance, topology_root)
    )
    validate_application_database_invariants(topology, binding.instance)
    return topology


def load_environment_topology(
    environment: str,
    topology_root: Path | None = None,
    *,
    profile_catalog_path: Path | None = None,
) -> ModelDeploymentTopology:
    """Compatibility wrapper for the explicit database-topology profile API."""
    return load_topology_profile(
        environment,
        topology_root,
        profile_catalog_path=profile_catalog_path,
    )


def validate_application_database_invariants(
    topology: ModelDeploymentTopology,
    topology_instance: str,
) -> None:
    """Fail on physical database, schema, role, or binding drift."""
    if topology_instance not in _EXPECTED_BINDING_DSN_ENVS:
        raise ValueError(
            f"Unsupported database topology instance '{topology_instance}'"
        )

    database = topology.databases.get(APPLICATION_DATABASE_REF)
    if database is None:
        raise ValueError("Topology must declare the 'application' database resource")
    if database.physical_name != APPLICATION_DATABASE_PHYSICAL_NAME:
        raise ValueError(
            "application database must resolve to "
            f"'{APPLICATION_DATABASE_PHYSICAL_NAME}', got '{database.physical_name}'"
        )

    actual_schemas = {name: schema.domain for name, schema in database.schemas.items()}
    if actual_schemas != _EXPECTED_SCHEMAS:
        raise ValueError(
            f"application schema/domain drift: expected {_EXPECTED_SCHEMAS}, "
            f"got {actual_schemas}"
        )
    actual_owners = {name: schema.owner for name, schema in database.schemas.items()}
    if actual_owners != _EXPECTED_SCHEMA_OWNERS:
        raise ValueError(
            f"application schema-owner drift: expected {_EXPECTED_SCHEMA_OWNERS}, "
            f"got {actual_owners}"
        )

    missing_principals = sorted(
        set(_EXPECTED_BINDING_PRINCIPALS.values()) - database.principals.keys()
    )
    if missing_principals:
        raise ValueError(f"application principals missing: {missing_principals}")

    if set(database.bindings) != set(_EXPECTED_BINDING_PRINCIPALS):
        raise ValueError(
            "application binding drift: expected "
            f"{sorted(_EXPECTED_BINDING_PRINCIPALS)}, got "
            f"{sorted(database.bindings)}"
        )
    for binding_name, expected_principal in _EXPECTED_BINDING_PRINCIPALS.items():
        binding = database.bindings[binding_name]
        if binding.database_ref != APPLICATION_DATABASE_REF:
            raise ValueError(
                f"Binding '{binding_name}' must resolve to database_ref "
                f"'{APPLICATION_DATABASE_REF}'"
            )
        if binding.principal != expected_principal:
            raise ValueError(
                f"Binding '{binding_name}' principal drift: expected "
                f"'{expected_principal}', got '{binding.principal}'"
            )
        expected_dsn_env = _EXPECTED_BINDING_DSN_ENVS[topology_instance][binding_name]
        if binding.dsn_env != expected_dsn_env:
            raise ValueError(
                f"Binding '{binding_name}' dsn_env drift: expected "
                f"'{expected_dsn_env}', got '{binding.dsn_env}'"
            )

    bound_principals = {binding.principal for binding in database.bindings.values()}
    if len(bound_principals) != len(database.bindings):
        raise ValueError(
            "Every application-domain pool must resolve to a distinct PostgreSQL "
            "principal"
        )
    if "omninode_runtime" not in topology.services:
        raise ValueError("Topology must declare the omninode_runtime service")
    if "omninode_runtime" in database.bindings:
        raise ValueError(
            "The omninode_runtime service name cannot double as a database binding; "
            "use omninode_runtime_service for the PostgreSQL principal namespace"
        )


def render_database_projection(
    environment: str,
    topology_root: Path | None = None,
    *,
    profile_catalog_path: Path | None = None,
) -> dict[str, object]:
    """Render the stable, secret-free database subset for downstream consumers."""
    binding = _resolve_profile(environment, profile_catalog_path)
    source_path = _topology_instance_path(binding.instance, topology_root)
    catalog_path = profile_catalog_path or _PROFILE_CATALOG_PATH
    topology = load_topology_profile(
        environment,
        topology_root,
        profile_catalog_path=profile_catalog_path,
    )
    dumped = topology.model_dump(mode="json")
    databases = cast("dict[str, object]", dumped["databases"])
    return {
        "schema_version": "1.0",
        "environment": environment,
        "topology_instance": binding.instance,
        "source": {
            "repository": _TOPOLOGY_REPOSITORY,
            "path": f"{_TOPOLOGY_SOURCE_PREFIX}/{binding.instance}.yaml",
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "profile_catalog_path": _PROFILE_CATALOG_SOURCE_PATH,
            "profile_catalog_sha256": hashlib.sha256(
                catalog_path.read_bytes()
            ).hexdigest(),
        },
        "databases": databases,
    }


def write_database_projection(
    environment: str,
    output: Path,
    topology_root: Path | None = None,
    *,
    profile_catalog_path: Path | None = None,
) -> None:
    """Write a deterministic database projection without secret material."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            render_database_projection(
                environment,
                topology_root,
                profile_catalog_path=profile_catalog_path,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def validate_database_projection(
    environment: str,
    projection_path: Path,
    topology_root: Path | None = None,
    *,
    profile_catalog_path: Path | None = None,
) -> None:
    """Require a checked-in projection to exactly match its typed source."""
    actual_raw = yaml.safe_load(projection_path.read_text(encoding="utf-8"))
    if not isinstance(actual_raw, dict):
        raise ValueError(f"Database projection must be a mapping: {projection_path}")
    actual = cast("dict[str, object]", actual_raw)
    expected = render_database_projection(
        environment,
        topology_root,
        profile_catalog_path=profile_catalog_path,
    )
    if actual != expected:
        raise ValueError(
            f"Database projection drift for '{environment}': {projection_path}; "
            "regenerate it from the checked-in typed topology"
        )


def _load_docker_consumer_catalog(path: Path) -> ModelDockerDatabaseConsumerCatalog:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelDockerDatabaseConsumerCatalog.model_validate(raw)


def validate_docker_topology_profile_injections(
    repo_root: Path = _REPOSITORY_ROOT,
    *,
    profile_catalog_path: Path | None = None,
) -> None:
    """Prove every checked-in Docker lane injects its exact DB profile."""
    catalog = _load_profile_catalog(profile_catalog_path)
    for profile, binding in catalog.profiles.items():
        if binding.deployment_repository != _TOPOLOGY_REPOSITORY:
            continue
        path = repo_root / binding.injection_path
        declared = set(_TOPOLOGY_PROFILE_LINE.findall(path.read_text(encoding="utf-8")))
        if profile not in declared:
            raise ValueError(
                f"Docker topology profile injection drift for '{profile}': "
                f"{binding.injection_path} must declare "
                f"{_TOPOLOGY_PROFILE_ENV_VAR}: {profile}"
            )

    runtime_policy_path = (
        repo_root / "contracts" / "services" / "runtime_policy.contract.yaml"
    )
    runtime_policy = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(runtime_policy_path.read_text(encoding="utf-8"))
    )
    mapped_runtime_profiles = {
        binding.runtime_policy_profile
        for binding in catalog.profiles.values()
        if binding.runtime_policy_profile is not None
    }
    if mapped_runtime_profiles != set(runtime_policy.profiles):
        raise ValueError("Docker topology runtime-policy profile coverage drift")


def validate_docker_catalog_parity(
    repo_root: Path = _REPOSITORY_ROOT,
    topology_root: Path | None = None,
) -> None:
    """Prove Docker DSN/database bindings consume the local typed topology."""
    environment = "local"
    topology = load_environment_topology(environment, topology_root)
    database = topology.databases[APPLICATION_DATABASE_REF]
    catalog = _load_docker_consumer_catalog(
        repo_root / "docker" / "catalog" / "database-consumers.yaml"
    )
    if catalog.environment != environment:
        raise ValueError(
            f"Docker database catalog targets '{catalog.environment}', expected 'local'"
        )

    services_dir = repo_root / "docker" / "catalog" / "services"
    application_dsn_envs = {binding.dsn_env for binding in database.bindings.values()}
    discovered_dsn_consumers: set[str] = set()
    for manifest_path in services_dir.glob("*.yaml"):
        manifest = _load_manifest(manifest_path)
        env_names = (
            set(manifest.required_env)
            | manifest.hardcoded_env.keys()
            | manifest.operational_defaults.keys()
            | manifest.catalog_env.keys()
        )
        if env_names & application_dsn_envs:
            discovered_dsn_consumers.add(manifest.name)

    declared_dsn_consumers = {
        service_name
        for service_name, consumer in catalog.consumers.items()
        if consumer.bindings
    } | set(catalog.deferred_consumers)
    if discovered_dsn_consumers != declared_dsn_consumers:
        raise ValueError(
            "Docker application-DSN inventory drift: discovered "
            f"{sorted(discovered_dsn_consumers)}, declared "
            f"{sorted(declared_dsn_consumers)}"
        )

    for service_name, consumer in catalog.consumers.items():
        manifest = _load_manifest(services_dir / f"{service_name}.yaml")
        env_values = {
            **manifest.hardcoded_env,
            **manifest.operational_defaults,
            **manifest.catalog_env,
        }
        env_names = set(manifest.required_env) | env_values.keys()
        for binding_name in consumer.bindings:
            binding = database.bindings.get(binding_name)
            if binding is None:
                raise ValueError(
                    f"Docker service '{service_name}' references unknown topology "
                    f"binding '{binding_name}'"
                )
            if binding.dsn_env not in env_names:
                raise ValueError(
                    f"Docker service '{service_name}' must consume "
                    f"{binding.dsn_env} for binding '{binding_name}'"
                )
        for env_name in consumer.physical_database_envs:
            if env_values.get(env_name) != database.physical_name:
                raise ValueError(
                    f"Docker service '{service_name}' {env_name} must be "
                    f"'{database.physical_name}', got {env_values.get(env_name)!r}"
                )


__all__ = [
    "APPLICATION_DATABASE_REF",
    "SUPPORTED_ENVIRONMENTS",
    "SUPPORTED_TOPOLOGY_PROFILES",
    "TOPOLOGY_PROFILE_INSTANCE_MAP",
    "load_environment_topology",
    "load_topology_profile",
    "render_database_projection",
    "validate_application_database_invariants",
    "validate_database_projection",
    "validate_docker_catalog_parity",
    "validate_docker_topology_profile_injections",
    "write_database_projection",
]
