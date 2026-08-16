# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed application-database targets shared by projection wiring tests."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
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
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.physical_schema_mapping import (
    physical_grant_schema_for_table,
)


@lru_cache(maxsize=1)
def application_topology() -> ModelDeploymentTopology:
    """Return the real shipped topology every runtime profile resolves.

    OMN-15656: this deliberately loads the platform instance rather than
    ``tests/fixtures/application_relation_ownership/topology.yaml``. The fixture
    topology carried hand-written TABLE grants that the shipped instances did
    not, so wiring tests proved a topology that does not exist and a 43/43
    strict-wiring failure shipped to onex-dev undetected.
    """
    return load_topology_profile("local")


ProjectionAccess = Literal["read", "write", "read_write"]


def _shipped_table_grant_exists(
    topology: ModelDeploymentTopology, principal: str, schema: str, table: str
) -> bool:
    """Return whether the shipped topology already grants ``table`` to ``principal``.

    Compares against the PHYSICAL grant schema
    (``physical_grant_schema_for_table``), not the caller-supplied logical
    schema: the shipped generator already applies the tenant/omninode_internal
    physical bridge when it writes TABLE grants, so a table pending its
    family's copy migration is checked-in with ``schema: public`` even though
    its logical domain is ``tenant``/``omninode_internal``. Comparing against
    the raw logical schema here would silently stop detecting an
    already-shipped grant for every bridged table -- exactly the false
    negative this helper exists to prevent.
    """
    database = topology.databases["application"]
    physical_schema = physical_grant_schema_for_table(schema, table)
    return any(
        grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == physical_schema
        and table in grant.objects
        for grant in database.principals[principal].grants
    )


def _topology_with_unshipped_grants(
    topology: ModelDeploymentTopology,
    tables: tuple[ModelDbTableDeclaration, ...],
    *,
    principal: str,
    privileges: tuple[EnumDatabasePrivilege, ...],
    reason: str,
) -> ModelDeploymentTopology:
    """Grant relations the platform deliberately does not ship yet.

    This is the ONLY sanctioned grant synthesis left in the test tree
    (OMN-15656). It exists for domains whose grants cannot be derived from node
    contracts — today only ``PLATFORM_CATALOG``, which requires a
    caller-supplied binding and which no ``db_io.db_tables`` block declares.

    It is fail-closed against the failure class this ticket fixed: synthesising
    a grant the shipped topology is *supposed* to carry is refused outright, so
    this helper can never be used to re-hide a missing platform grant the way
    the old ``_with_projection_fixture_grants`` did.
    """
    if not reason.strip():
        raise ValueError("unshipped grant synthesis requires an explicit reason")
    database = topology.databases["application"]
    for table in tables:
        if _shipped_table_grant_exists(topology, principal, table.schema, table.name):
            raise AssertionError(
                f"{table.schema}.{table.name} is already granted to {principal!r} by "
                "the shipped topology; use the real grant instead of synthesising one"
            )
    principals = dict(database.principals)
    target = principals[principal]
    principals[principal] = target.model_copy(
        update={
            "grants": (
                *target.grants,
                *(
                    ModelDeploymentTopologyDatabaseGrant(
                        object_type=EnumDatabaseGrantObjectType.TABLE,
                        schema=table.schema,
                        objects=(table.name,),
                        privileges=privileges,
                    )
                    for table in tables
                ),
            )
        }
    )
    return topology.model_copy(
        update={
            "databases": {
                "application": database.model_copy(update={"principals": principals})
            }
        }
    )


def projection_database_target(
    *table_names: str,
    schema: str = "tenant",
    physical_database: str = "omnidash_analytics",
    access: ProjectionAccess = "read_write",
    catalog_read_binding: str | None = None,
    catalog_write_binding: str | None = None,
    unshipped_grant_principal: str | None = None,
    unshipped_grant_reason: str = "",
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
    if unshipped_grant_principal is not None:
        topology = _topology_with_unshipped_grants(
            topology,
            tables,
            principal=unshipped_grant_principal,
            privileges=(
                (EnumDatabasePrivilege.SELECT,)
                if access == "read"
                else (
                    EnumDatabasePrivilege.SELECT,
                    EnumDatabasePrivilege.INSERT,
                    EnumDatabasePrivilege.UPDATE,
                )
            ),
            reason=unshipped_grant_reason,
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
