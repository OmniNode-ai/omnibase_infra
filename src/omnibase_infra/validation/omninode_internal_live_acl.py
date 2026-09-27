# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17886: the live ``omninode_internal`` ACLs, diffed against the topology.

Every other grant gate in this repository checks DECLARATIONS: that the
topology renders to the catalog it claims (``--check --prove``), or that a
migration in the corpus issues each declared grant
(``check_topology_grant_delivery``). Neither reads a live database, so neither
can see what OMN-17886 measured on 2026-09-04: onex-dev denying
``omninode_runtime`` two tables it declares, while the ``.201`` lane grants it
three it never declared, through a schema-wide ``ALTER DEFAULT PRIVILEGES``.

This module reads the live catalog -- relation ACLs (``pg_class.relacl``),
the schema ACL (``pg_namespace.nspacl``), default-privilege rules
(``pg_default_acl``) and relation owners -- for ONE schema, and diffs it in
both directions against the resolved topology declaration:

* an undeclared grant (live, not declared) is a finding;
* a declared grant that is missing live is a finding;
* a declared relation that does not exist in the schema is a finding;
* a default-privilege rule in the schema is a finding, because the typed
  topology has no way to declare one (OMN-15809);
* relation owners are reported but are not findings: the owner question is
  OMN-17886 AC3, settled by readback, not by this gate.

A relation's own owner is excluded as a grantee: its implicit full privileges
are ownership, not a grant.

"Resolved" means through :func:`physical_grant_schema_for_table`: a table the
topology declares against ``omninode_internal`` but which physically still
lives in ``public`` (the ``INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359``
bridge) is not expected in this schema, so it is neither required nor
reported here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.models.core import ModelDeploymentTopology
from omnibase_infra.topology.physical_schema_mapping import (
    physical_grant_schema_for_table,
)

__all__ = [
    "DEFAULT_SCHEMA",
    "LIVE_DEFAULT_ACL_QUERY",
    "LIVE_RELATION_ACL_QUERY",
    "LIVE_SCHEMA_ACL_QUERY",
    "DeclaredAcl",
    "LiveAclReport",
    "declared_acl",
    "diff_live_acl",
]

DEFAULT_SCHEMA = "omninode_internal"

# Parameterised by schema name. Mirrors scripts/apply_application_database_acl.py
# (_RELATION_ACL_QUERY / _DEFAULT_ACL_QUERY / _SCHEMA_ACL_QUERY) narrowed to one
# schema, so the two readers cannot disagree about how an ACL is exploded.
LIVE_RELATION_ACL_QUERY = """
SELECT relation.relname AS relation_name,
       relation.relkind,
       owner.rolname AS owner,
       COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
       acl.privilege_type
FROM pg_class relation
JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
JOIN pg_roles owner ON owner.oid = relation.relowner
CROSS JOIN LATERAL aclexplode(
    COALESCE(relation.relacl, acldefault('r', relation.relowner))
) acl
LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
WHERE namespace.nspname = %s
  AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY relation_name, grantee, privilege_type
"""

LIVE_SCHEMA_ACL_QUERY = """
SELECT owner.rolname AS owner,
       COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
       acl.privilege_type
FROM pg_namespace namespace
JOIN pg_roles owner ON owner.oid = namespace.nspowner
CROSS JOIN LATERAL aclexplode(
    COALESCE(namespace.nspacl, acldefault('n', namespace.nspowner))
) acl
LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
WHERE namespace.nspname = %s
ORDER BY grantee, privilege_type
"""

LIVE_DEFAULT_ACL_QUERY = """
SELECT owner.rolname AS owner,
       default_acl.defaclobjtype AS object_type,
       COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
       acl.privilege_type
FROM pg_default_acl default_acl
JOIN pg_roles owner ON owner.oid = default_acl.defaclrole
JOIN pg_namespace namespace ON namespace.oid = default_acl.defaclnamespace
CROSS JOIN LATERAL aclexplode(default_acl.defaclacl) acl
LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
WHERE namespace.nspname = %s
ORDER BY owner, object_type, grantee, privilege_type
"""


@dataclass(frozen=True, slots=True)
class DeclaredAcl:
    """What the topology declares for one schema, physically resolved."""

    schema: str
    # (principal, relation, privilege)
    table_grants: frozenset[tuple[str, str, str]]
    # (principal, privilege) on the schema itself
    schema_grants: frozenset[tuple[str, str]]

    @property
    def relations(self) -> frozenset[str]:
        return frozenset(relation for _, relation, _ in self.table_grants)


@dataclass(frozen=True, slots=True)
class LiveAclReport:
    """The two-way difference between the live catalog and the declaration."""

    schema: str
    undeclared_table_grants: tuple[tuple[str, str, str], ...]
    missing_table_grants: tuple[tuple[str, str, str], ...]
    missing_relations: tuple[str, ...]
    undeclared_schema_grants: tuple[tuple[str, str], ...]
    missing_schema_grants: tuple[tuple[str, str], ...]
    # (owner, object_type, grantee, privilege)
    default_privilege_rules: tuple[tuple[str, str, str, str], ...]
    # relation -> owner, informational (OMN-17886 AC3)
    relation_owners: Mapping[str, str] = field(default_factory=dict)
    observed_relation_count: int = 0

    @property
    def findings(self) -> tuple[str, ...]:
        lines: list[str] = []
        lines += [
            f"UNDECLARED_GRANT {p} {priv} ON {self.schema}.{r}"
            for p, r, priv in self.undeclared_table_grants
        ]
        lines += [
            f"MISSING_DECLARED_GRANT {p} {priv} ON {self.schema}.{r}"
            for p, r, priv in self.missing_table_grants
        ]
        lines += [
            f"MISSING_DECLARED_RELATION {self.schema}.{r}"
            for r in self.missing_relations
        ]
        lines += [
            f"UNDECLARED_SCHEMA_GRANT {p} {priv} ON SCHEMA {self.schema}"
            for p, priv in self.undeclared_schema_grants
        ]
        lines += [
            f"MISSING_DECLARED_SCHEMA_GRANT {p} {priv} ON SCHEMA {self.schema}"
            for p, priv in self.missing_schema_grants
        ]
        lines += [
            f"UNDECLARED_DEFAULT_PRIVILEGE owner={o} objtype={t} {g} {priv} "
            f"IN SCHEMA {self.schema}"
            for o, t, g, priv in self.default_privilege_rules
        ]
        return tuple(lines)

    @property
    def clean(self) -> bool:
        return not self.findings


def declared_acl(
    topology: ModelDeploymentTopology,
    *,
    database_ref: str = "application",
    schema: str = DEFAULT_SCHEMA,
) -> DeclaredAcl:
    """Resolve what ``topology`` declares for ``schema``, physically.

    A TABLE grant counts only for objects whose physical schema, per
    :func:`physical_grant_schema_for_table`, is ``schema`` itself.
    """
    database = topology.databases.get(database_ref)
    if database is None:
        msg = f"topology declares no database_ref {database_ref!r}"
        raise ValueError(msg)
    table_grants: set[tuple[str, str, str]] = set()
    schema_grants: set[tuple[str, str]] = set()
    for principal, declaration in database.principals.items():
        for grant in declaration.grants:
            if grant.schema != schema:
                continue
            privileges = {str(privilege) for privilege in grant.privileges}
            if grant.object_type == EnumDatabaseGrantObjectType.SCHEMA:
                schema_grants.update((principal, p) for p in privileges)
            elif grant.object_type == EnumDatabaseGrantObjectType.TABLE:
                for relation in grant.objects:
                    if physical_grant_schema_for_table(schema, relation) != schema:
                        continue
                    table_grants.update((principal, relation, p) for p in privileges)
    return DeclaredAcl(
        schema=schema,
        table_grants=frozenset(table_grants),
        schema_grants=frozenset(schema_grants),
    )


def diff_live_acl(
    declared: DeclaredAcl,
    *,
    relation_rows: Iterable[Mapping[str, object]],
    schema_rows: Iterable[Mapping[str, object]],
    default_rows: Iterable[Mapping[str, object]],
) -> LiveAclReport:
    """Diff live catalog rows (shaped as the LIVE_*_QUERY results) against
    ``declared``. Pure: no database access, so it is unit-testable."""
    owners: dict[str, str] = {}
    live_table: set[tuple[str, str, str]] = set()
    for row in relation_rows:
        relation = str(row["relation_name"])
        owner = str(row["owner"])
        owners[relation] = owner
        grantee = str(row["grantee"])
        if grantee == owner:
            continue
        live_table.add((grantee, relation, str(row["privilege_type"])))

    live_schema: set[tuple[str, str]] = set()
    for row in schema_rows:
        grantee = str(row["grantee"])
        if grantee == str(row["owner"]):
            continue
        live_schema.add((grantee, str(row["privilege_type"])))

    defaults = tuple(
        sorted(
            (
                str(row["owner"]),
                str(row["object_type"]),
                str(row["grantee"]),
                str(row["privilege_type"]),
            )
            for row in default_rows
        )
    )

    missing_relations = tuple(sorted(declared.relations - set(owners)))
    # A grant on a relation that does not exist is reported once, as the
    # missing relation, not again once per privilege.
    missing_grants = {
        grant
        for grant in declared.table_grants - live_table
        if grant[1] not in missing_relations
    }
    return LiveAclReport(
        schema=declared.schema,
        undeclared_table_grants=tuple(sorted(live_table - declared.table_grants)),
        missing_table_grants=tuple(sorted(missing_grants)),
        missing_relations=missing_relations,
        undeclared_schema_grants=tuple(sorted(live_schema - declared.schema_grants)),
        missing_schema_grants=tuple(sorted(declared.schema_grants - live_schema)),
        default_privilege_rules=defaults,
        relation_owners=dict(sorted(owners.items())),
        observed_relation_count=len(owners),
    )
