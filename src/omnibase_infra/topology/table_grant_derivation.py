# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Derive ``object_type: TABLE`` grants from node contract ``db_io.db_tables``.

OMN-15656. The OMN-15418 privilege validator
(:func:`omnibase_infra.runtime.auto_wiring.handler_wiring._require_projection_binding_privileges`)
requires an explicit per-table grant before a projection handler may wire. The
checked-in topology instances declared none, so every contract-declared table
failed on every profile.

This module is the *authoring* side of that contract: the grants are a
projection of the node contracts that consume them, never a hand-maintained
list. Contracts own ``schema``/``name``/``access``; the deployment topology owns
which principal serves which schema domain. Nothing here invents a relation, and
nothing here widens a privilege beyond what the validator demands.

The derivation is deliberately total and fail-closed: a declaration this module
cannot map to a topology principal is returned as a typed residual with a
reason, never silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_database_grant import (
    ModelDeploymentTopologyDatabaseGrant,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _INTERNAL_PROJECTION_BINDING,
    _TENANT_PROJECTION_BINDING,
)
from omnibase_infra.topology.physical_schema_mapping import (
    TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    physical_grant_schema_for_table,
)

__all__ = [
    "ContractTableDeclaration",
    "DerivedTableGrants",
    "UnmappableDeclaration",
    "READ_PRIVILEGES",
    "WRITE_PRIVILEGES",
    "DOMAIN_PROJECTION_BINDINGS",
    "TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359",
    "physical_grant_schema_for_table",
    "derive_table_grants",
    "load_contract_declarations",
]

# Mirrors ``_require_projection_binding_privileges``: PostgreSQL needs SELECT
# alongside INSERT/UPDATE because the projection adapter issues
# ``INSERT ... ON CONFLICT DO UPDATE``.
READ_PRIVILEGES: frozenset[EnumDatabasePrivilege] = frozenset(
    {EnumDatabasePrivilege.SELECT}
)
WRITE_PRIVILEGES: frozenset[EnumDatabasePrivilege] = frozenset(
    {
        EnumDatabasePrivilege.SELECT,
        EnumDatabasePrivilege.INSERT,
        EnumDatabasePrivilege.UPDATE,
    }
)

# Mirrors ``_projection_operation_bindings``. PLATFORM_CATALOG is absent by
# design: that domain requires a caller-supplied read/write binding, so a
# contract declaration alone cannot name the principal. Such declarations are
# returned as residuals instead of being guessed at.
DOMAIN_PROJECTION_BINDINGS: Mapping[EnumDatabaseSchemaDomain, str] = {
    EnumDatabaseSchemaDomain.TENANT: _TENANT_PROJECTION_BINDING,
    EnumDatabaseSchemaDomain.OMNINODE_INTERNAL: _INTERNAL_PROJECTION_BINDING,
}


@dataclass(frozen=True, slots=True)
class ContractTableDeclaration:
    """One ``db_io.db_tables`` entry tied back to the contract that declared it."""

    node: str
    contract_path: Path
    table: ModelDbTableDeclaration


@dataclass(frozen=True, slots=True)
class UnmappableDeclaration:
    """A declaration that cannot be projected onto a topology principal."""

    node: str
    database_ref: str
    schema: str
    name: str
    reason: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Stable identity used by the shrink-only residual ratchet."""
        return (self.database_ref, self.schema, self.name)


@dataclass(frozen=True, slots=True)
class DerivedTableGrants:
    """TABLE grants per principal plus the residuals that could not be derived."""

    grants: Mapping[str, tuple[ModelDeploymentTopologyDatabaseGrant, ...]]
    unmappable: tuple[UnmappableDeclaration, ...]


def _privileges_for_access(access: str) -> frozenset[EnumDatabasePrivilege]:
    """Map a declared access mode onto the privileges the validator demands."""
    if access == "read":
        return READ_PRIVILEGES
    if access in {"write", "read_write"}:
        # read_write is the union, and WRITE_PRIVILEGES already contains SELECT.
        return WRITE_PRIVILEGES
    raise ValueError(f"Unsupported db_tables access mode {access!r}")


def load_contract_declarations(
    contracts_root: Path,
) -> tuple[ContractTableDeclaration, ...]:
    """Read every ``contract.yaml`` under ``contracts_root`` for ``db_io.db_tables``.

    Raises when the root does not exist so a missing cross-repo checkout fails
    the gate loudly instead of degrading into a vacuous zero-declaration pass.
    """
    if not contracts_root.is_dir():
        raise FileNotFoundError(
            f"contracts root {contracts_root} does not exist; the cross-repo "
            "checkout that provides node contracts is required for derivation"
        )
    declarations: list[ContractTableDeclaration] = []
    for contract_path in sorted(contracts_root.rglob("contract.yaml")):
        document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        db_io = document.get("db_io")
        if not isinstance(db_io, dict):
            continue
        entries = db_io.get("db_tables")
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            declarations.append(
                ContractTableDeclaration(
                    node=contract_path.parent.name,
                    contract_path=contract_path,
                    table=ModelDbTableDeclaration(**entry),
                )
            )
    if not declarations:
        raise ValueError(
            f"no db_io.db_tables declarations found under {contracts_root}; "
            "refusing to derive an empty grant set"
        )
    return tuple(declarations)


def derive_table_grants(
    topology: ModelDeploymentTopology,
    declarations: Iterable[ContractTableDeclaration],
    *,
    database_ref: str = "application",
) -> DerivedTableGrants:
    """Project contract table declarations onto per-principal TABLE grants."""
    database = topology.databases.get(database_ref)
    if database is None:
        raise ValueError(f"topology declares no database {database_ref!r}")

    # (principal, schema, table) -> required privileges, unioned across every
    # contract that declares the same relation.
    required: dict[tuple[str, str, str], set[EnumDatabasePrivilege]] = {}
    unmappable: dict[tuple[str, str, str], UnmappableDeclaration] = {}

    for declaration in declarations:
        table = declaration.table
        key = (table.database_ref, table.schema, table.name)
        if table.database_ref != database_ref:
            unmappable.setdefault(
                key,
                UnmappableDeclaration(
                    node=declaration.node,
                    database_ref=table.database_ref,
                    schema=table.schema,
                    name=table.name,
                    reason=(
                        f"database_ref {table.database_ref!r} is not declared in "
                        "the application topology"
                    ),
                ),
            )
            continue
        schema = database.schemas.get(table.schema)
        if schema is None:
            unmappable.setdefault(
                key,
                UnmappableDeclaration(
                    node=declaration.node,
                    database_ref=table.database_ref,
                    schema=table.schema,
                    name=table.name,
                    reason=f"schema {table.schema!r} is not declared in the topology",
                ),
            )
            continue
        binding_ref = DOMAIN_PROJECTION_BINDINGS.get(schema.domain)
        if binding_ref is None:
            unmappable.setdefault(
                key,
                UnmappableDeclaration(
                    node=declaration.node,
                    database_ref=table.database_ref,
                    schema=table.schema,
                    name=table.name,
                    reason=(
                        f"domain {schema.domain.value} requires an explicit "
                        "caller-supplied binding and cannot be derived from a "
                        "contract declaration alone"
                    ),
                ),
            )
            continue
        binding = database.bindings.get(binding_ref)
        if binding is None:
            unmappable.setdefault(
                key,
                UnmappableDeclaration(
                    node=declaration.node,
                    database_ref=table.database_ref,
                    schema=table.schema,
                    name=table.name,
                    reason=f"topology declares no binding {binding_ref!r}",
                ),
            )
            continue
        grant_schema = physical_grant_schema_for_table(table.schema, table.name)
        entry = required.setdefault(
            (binding.principal, grant_schema, table.name), set()
        )
        entry.update(_privileges_for_access(table.access))

    grants: dict[str, tuple[ModelDeploymentTopologyDatabaseGrant, ...]] = {}
    # Group by (principal, schema, privilege set) so each principal carries at
    # most one grant per schema per distinct privilege shape.
    grouped: dict[str, dict[tuple[str, tuple[str, ...]], list[str]]] = {}
    for (principal, schema_name, table_name), privileges in required.items():
        privilege_key = tuple(sorted(privilege.value for privilege in privileges))
        grouped.setdefault(principal, {}).setdefault(
            (schema_name, privilege_key), []
        ).append(table_name)

    for principal in sorted(grouped):
        principal_grants: list[ModelDeploymentTopologyDatabaseGrant] = []
        for schema_name, privilege_key in sorted(grouped[principal]):
            names = sorted(grouped[principal][(schema_name, privilege_key)])
            principal_grants.append(
                ModelDeploymentTopologyDatabaseGrant(
                    object_type=EnumDatabaseGrantObjectType.TABLE,
                    schema=schema_name,
                    objects=tuple(names),
                    privileges=tuple(
                        EnumDatabasePrivilege(value) for value in privilege_key
                    ),
                )
            )
        grants[principal] = tuple(principal_grants)

    return DerivedTableGrants(
        grants=grants,
        unmappable=tuple(
            unmappable[key] for key in sorted(unmappable, key=lambda item: item)
        ),
    )
