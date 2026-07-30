# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed application-database targets shared by projection wiring tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_database_grant import (
    ModelDeploymentTopologyDatabaseGrant,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDatabaseTarget,
    _resolve_projection_database_target,
)

_TOPOLOGY_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "application_relation_ownership"
    / "topology.yaml"
)


@lru_cache(maxsize=1)
def application_topology() -> ModelDeploymentTopology:
    return ModelDeploymentTopology.from_yaml(_TOPOLOGY_PATH)


ProjectionAccess = Literal["read", "write", "read_write"]


def _with_projection_fixture_grants(
    topology: ModelDeploymentTopology,
    tables: tuple[ModelDbTableDeclaration, ...],
    *,
    catalog_read_binding: str | None,
    catalog_write_binding: str | None,
) -> ModelDeploymentTopology:
    """Add exact per-table grants requested by a focused test fixture."""
    database = topology.databases["application"]
    required: dict[tuple[str, str, str], set[EnumDatabasePrivilege]] = {}
    for table in tables:
        domain = topology.schema_domain(table.database_ref, table.schema)
        if domain is EnumDatabaseSchemaDomain.TENANT:
            read_ref = write_ref = "tenant_projection"
        elif domain is EnumDatabaseSchemaDomain.OMNINODE_INTERNAL:
            read_ref = write_ref = "omninode_runtime_service"
        else:
            read_ref = catalog_read_binding
            write_ref = catalog_write_binding
        operations: tuple[tuple[str | None, set[EnumDatabasePrivilege]], ...] = (
            (
                read_ref if table.access in {"read", "read_write"} else None,
                {EnumDatabasePrivilege.SELECT},
            ),
            (
                write_ref if table.access in {"write", "read_write"} else None,
                {
                    EnumDatabasePrivilege.SELECT,
                    EnumDatabasePrivilege.INSERT,
                    EnumDatabasePrivilege.UPDATE,
                },
            ),
        )
        for binding_ref, privileges in operations:
            binding = database.bindings.get(binding_ref) if binding_ref else None
            if binding is None:
                continue
            key = (binding.principal, table.schema, table.name)
            required.setdefault(key, set()).update(privileges)

    principals = dict(database.principals)
    for (principal_name, schema, table_name), privileges in required.items():
        principal = principals[principal_name]
        grants = list(principal.grants)
        has_schema_usage = any(
            grant.object_type is EnumDatabaseGrantObjectType.SCHEMA
            and grant.schema == schema
            and EnumDatabasePrivilege.USAGE in grant.privileges
            for grant in grants
        )
        if not has_schema_usage:
            grants.append(
                ModelDeploymentTopologyDatabaseGrant(
                    object_type=EnumDatabaseGrantObjectType.SCHEMA,
                    schema=schema,
                    privileges=(EnumDatabasePrivilege.USAGE,),
                )
            )
        already_granted = {
            privilege
            for grant in grants
            if grant.object_type is EnumDatabaseGrantObjectType.TABLE
            and grant.schema == schema
            and table_name in grant.objects
            for privilege in grant.privileges
        }
        missing = tuple(
            sorted(privileges - already_granted, key=lambda item: item.value)
        )
        if missing:
            grants.append(
                ModelDeploymentTopologyDatabaseGrant(
                    object_type=EnumDatabaseGrantObjectType.TABLE,
                    schema=schema,
                    objects=(table_name,),
                    privileges=missing,
                )
            )
        principals[principal_name] = principal.model_copy(
            update={"grants": tuple(grants)}
        )
    database = database.model_copy(update={"principals": principals})
    return topology.model_copy(update={"databases": {"application": database}})


def projection_database_target(
    *table_names: str,
    schema: str = "tenant",
    physical_database: str = "omnidash_analytics",
    access: ProjectionAccess = "read_write",
    catalog_read_binding: str | None = None,
    catalog_write_binding: str | None = None,
) -> ProjectionDatabaseTarget:
    names = table_names or ("projection_fixture",)
    tables = tuple(
        ModelDbTableDeclaration(
            name=name,
            database_ref="application",
            schema=schema,
            migration=f"tests/{name}.sql",
            access=access,
            role=f"{name}_projection",
        )
        for name in names
    )
    topology = application_topology()
    topology = _with_projection_fixture_grants(
        topology,
        tables,
        catalog_read_binding=catalog_read_binding,
        catalog_write_binding=catalog_write_binding,
    )
    if physical_database != topology.databases["application"].physical_name:
        database = topology.databases["application"].model_copy(
            update={"physical_name": physical_database}
        )
        topology = topology.model_copy(update={"databases": {"application": database}})
    return _resolve_projection_database_target(
        tables,
        topology,
        catalog_read_binding=catalog_read_binding,
        catalog_write_binding=catalog_write_binding,
    )


def projection_database_urls(
    target: ProjectionDatabaseTarget,
    default_url: str,
    **binding_urls: str,
) -> dict[str, str]:
    """Build an exact binding→DSN map for focused adapter tests."""
    return {
        binding.binding_ref: binding_urls.get(binding.binding_ref, default_url)
        for binding in target.bindings
    }


__all__ = [
    "application_topology",
    "projection_database_target",
    "projection_database_urls",
]
