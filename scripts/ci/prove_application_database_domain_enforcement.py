# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rebuilt PostgreSQL 16 proof for application-domain enforcement gates."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import UUID

import psycopg2
from psycopg2 import sql as pg_sql

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_function_definition_sha256,
    lint_application_database_sql,
    load_application_database_ownership_identities,
    validate_application_database_catalog_census,
    validate_application_database_pool_identities,
    validate_application_database_relation_states,
)
from omnibase_infra.validation.application_database_source_tenant_authority import (
    resolve_application_database_authority_columns,
)
from omnibase_infra.validation.application_relation_ownership import (
    load_service_ownership_manifest,
    validate_application_relation_ownership,
)
from omnibase_infra.validation.enums.enum_application_database_identity_root_operation import (
    EnumApplicationDatabaseIdentityRootOperation,
)
from omnibase_infra.validation.enums.enum_application_inventory_object_kind import (
    EnumApplicationInventoryObjectKind,
)
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)
from omnibase_infra.validation.models.model_application_database_catalog_identity import (
    ModelApplicationDatabaseCatalogIdentity,
)
from omnibase_infra.validation.models.model_application_database_column_state import (
    ModelApplicationDatabaseColumnState,
)
from omnibase_infra.validation.models.model_application_database_function_state import (
    ModelApplicationDatabaseFunctionState,
)
from omnibase_infra.validation.models.model_application_database_identity_root_control_state import (
    ModelApplicationDatabaseIdentityRootControlState,
)
from omnibase_infra.validation.models.model_application_database_policy_state import (
    ModelApplicationDatabasePolicyState,
)
from omnibase_infra.validation.models.model_application_database_pool_identity import (
    ModelApplicationDatabasePoolIdentity,
)
from omnibase_infra.validation.models.model_application_database_relation_state import (
    ModelApplicationDatabaseRelationState,
)
from omnibase_infra.validation.models.model_application_database_routine_dependency_state import (
    ModelApplicationDatabaseRoutineDependencyState,
)
from omnibase_infra.validation.models.model_application_database_tenant_isolation_evidence import (
    ModelApplicationDatabaseTenantIsolationEvidence,
)
from omnibase_infra.validation.models.model_application_relation_declaration import (
    ModelApplicationRelationDeclaration,
)
from omnibase_infra.validation.models.model_application_relation_inventory import (
    ModelApplicationRelationInventory,
)
from omnibase_infra.validation.models.model_database_object_evidence import (
    ModelDatabaseObjectEvidence,
)
from omnibase_infra.validation.models.model_live_application_relation import (
    ModelLiveApplicationRelation,
)
from omnibase_infra.validation.models.model_relation_evidence import (
    ModelRelationEvidence,
)

_ADMIN_DSN = os.environ["ADMIN_DSN"]
_OWNERSHIP_MANIFEST = Path(os.environ["OWNERSHIP_MANIFEST"])
_OWNERSHIP = load_service_ownership_manifest(_OWNERSHIP_MANIFEST)
_TOPOLOGY = load_topology_profile("local")
_DATABASE = _TOPOLOGY.databases["application"]
_TENANT_A = UUID("11111111-1111-1111-1111-111111111111")
_TENANT_B = UUID("22222222-2222-2222-2222-222222222222")
_EXPECTED_TENANT_ROWS = {_TENANT_A: 2, _TENANT_B: 1}
_POLICY_COMMANDS = {
    "*": "ALL",
    "r": "SELECT",
    "a": "INSERT",
    "w": "UPDATE",
    "d": "DELETE",
}
_CATALOG_TO_RELATION_KIND = {
    EnumApplicationInventoryObjectKind.TABLE: EnumApplicationRelationKind.TABLE,
    EnumApplicationInventoryObjectKind.VIEW: EnumApplicationRelationKind.VIEW,
    EnumApplicationInventoryObjectKind.MATERIALIZED_VIEW: (
        EnumApplicationRelationKind.MATERIALIZED_VIEW
    ),
    EnumApplicationInventoryObjectKind.FOREIGN_TABLE: (
        EnumApplicationRelationKind.FOREIGN_TABLE
    ),
    EnumApplicationInventoryObjectKind.FUNCTION: EnumApplicationRelationKind.FUNCTION,
}


def _pool_dsn(binding_ref: str) -> str:
    """Resolve proof plumbing from a topology-derived binding name."""
    return os.environ[f"POOL_DSN_{binding_ref.upper()}"]


def _connect(dsn: str) -> psycopg2.extensions.connection:
    return psycopg2.connect(dsn)


def _relation_evidence(
    declaration: ModelApplicationRelationDeclaration,
) -> ModelRelationEvidence:
    """Resolve one manifest classification for a relation identity."""
    matches = tuple(
        evidence
        for evidence in _OWNERSHIP.relation_evidence
        if evidence.database_ref == declaration.database_ref
        and evidence.schema == declaration.schema
        and evidence.name == declaration.name
        and evidence.kind is declaration.kind
        and evidence.function_signature == declaration.function_signature
    )
    if len(matches) != 1:
        raise AssertionError(
            f"{declaration.identity!r} requires exactly one typed relation "
            f"classification; observed {len(matches)}"
        )
    return matches[0]


def _database_object_evidence(
    declaration: ModelApplicationRelationDeclaration,
) -> ModelDatabaseObjectEvidence:
    """Resolve one independently authored database-object audit declaration."""
    matches = tuple(
        evidence
        for evidence in _OWNERSHIP.database_objects
        if evidence.database_ref == declaration.database_ref
        and evidence.schema == declaration.schema
        and evidence.name == declaration.name
        and evidence.kind.value == declaration.kind.value
        and evidence.function_signature == declaration.function_signature
    )
    if len(matches) != 1:
        raise AssertionError(
            f"{declaration.identity!r} requires exactly one typed database-object "
            f"audit; observed {len(matches)}"
        )
    return matches[0]


def _column_states(
    connection: psycopg2.extensions.connection,
    schema: str,
    name: str,
) -> tuple[ModelApplicationDatabaseColumnState, ...]:
    """Read exact columns for tables, views, materialized views, and foreign tables."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT attribute.attname,
                   format_type(attribute.atttypid, attribute.atttypmod),
                   NOT attribute.attnotnull,
                   CASE WHEN attribute.attgenerated = ''
                     THEN pg_get_expr(default_row.adbin, default_row.adrelid)
                   END,
                   CASE WHEN attribute.attgenerated <> ''
                     THEN pg_get_expr(default_row.adbin, default_row.adrelid)
                   END
            FROM pg_attribute attribute
            JOIN pg_class relation ON relation.oid = attribute.attrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            LEFT JOIN pg_attrdef default_row
              ON default_row.adrelid = attribute.attrelid
             AND default_row.adnum = attribute.attnum
            WHERE namespace.nspname = %s
              AND relation.relname = %s
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            ORDER BY attribute.attnum
            """,
            (schema, name),
        )
        return tuple(
            ModelApplicationDatabaseColumnState(
                name=row[0],
                data_type=row[1],
                nullable=row[2],
                default_expression=row[3],
                generated_expression=row[4],
            )
            for row in cursor.fetchall()
        )


def _prove_identity_root_enumeration(
    *,
    schema: str,
    name: str,
    role: str,
) -> bool:
    relation = pg_sql.SQL("{}.{}").format(
        pg_sql.Identifier(schema),
        pg_sql.Identifier(name),
    )
    admin = _connect(_ADMIN_DSN)
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                pg_sql.SQL("SET LOCAL ROLE {}").format(pg_sql.Identifier(role))
            )
            cursor.execute(pg_sql.SQL("SELECT count(*) FROM {}").format(relation))
            control_count = int(cursor.fetchone()[0])
        admin.rollback()
    finally:
        admin.close()

    runtime = _connect(_pool_dsn("onex_api"))
    try:
        with runtime.cursor() as cursor:
            cursor.execute("SET LOCAL app.tenant_id = %s", (str(_TENANT_A),))
            cursor.execute(pg_sql.SQL("SELECT count(*) FROM {}").format(relation))
            tenant_count = int(cursor.fetchone()[0])
        runtime.rollback()
    finally:
        runtime.close()

    unset_runtime = _connect(_pool_dsn("onex_api"))
    try:
        with unset_runtime.cursor() as cursor:
            cursor.execute(pg_sql.SQL("SELECT count(*) FROM {}").format(relation))
            unset_count = int(cursor.fetchone()[0])
        unset_runtime.rollback()
    finally:
        unset_runtime.close()
    return control_count == 2 and tenant_count == 1 and unset_count == 0


def _prove_identity_root_creation(
    *,
    schema: str,
    name: str,
    role: str,
) -> bool:
    relation = pg_sql.SQL("{}.{}").format(
        pg_sql.Identifier(schema),
        pg_sql.Identifier(name),
    )
    new_tenant = UUID("33333333-3333-3333-3333-333333333333")
    admin = _connect(_ADMIN_DSN)
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                pg_sql.SQL("SET LOCAL ROLE {}").format(pg_sql.Identifier(role))
            )
            cursor.execute(
                pg_sql.SQL("INSERT INTO {} (id, tenant_name) VALUES (%s, %s)").format(
                    relation
                ),
                (str(new_tenant), "control-created"),
            )
        control_inserted = True
        admin.rollback()
    except psycopg2.Error:
        admin.rollback()
        control_inserted = False
    finally:
        admin.close()

    runtime = _connect(_pool_dsn("onex_api"))
    try:
        try:
            with runtime.cursor() as cursor:
                cursor.execute("SET LOCAL app.tenant_id = %s", (str(_TENANT_A),))
                cursor.execute(
                    pg_sql.SQL(
                        "INSERT INTO {} (id, tenant_name) VALUES (%s, %s)"
                    ).format(relation),
                    (str(new_tenant), "runtime-cross-tenant"),
                )
        except psycopg2.Error as exc:
            runtime.rollback()
            runtime_denied = str(exc.pgcode) == "42501"
        else:
            runtime.rollback()
            runtime_denied = False
    finally:
        runtime.close()
    return control_inserted and runtime_denied


def _runtime_identity_root_membership_principals(
    connection: psycopg2.extensions.connection,
    role: str,
) -> tuple[str, ...]:
    """Return runtime roles with a zero-hop, direct, or transitive path to role."""
    runtime_principals = sorted(
        binding.principal for binding in _DATABASE.bindings.values()
    )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            WITH RECURSIVE runtime_role_path AS (
                SELECT runtime_role.oid AS root_member,
                       runtime_role.oid AS reachable_role,
                       ARRAY[runtime_role.oid]::oid[] AS path
                FROM pg_roles AS runtime_role
                WHERE runtime_role.rolname = ANY(%s)

                UNION ALL

                SELECT prior.root_member,
                       membership.roleid AS reachable_role,
                       prior.path || membership.roleid
                FROM runtime_role_path AS prior
                JOIN pg_auth_members AS membership
                  ON membership.member = prior.reachable_role
                WHERE NOT membership.roleid = ANY(prior.path)
            )
            SELECT DISTINCT runtime_role.rolname
            FROM runtime_role_path AS reachable
            JOIN pg_roles AS runtime_role
              ON runtime_role.oid = reachable.root_member
            JOIN pg_roles AS reached_role
              ON reached_role.oid = reachable.reachable_role
            WHERE reached_role.rolname = %s
            ORDER BY runtime_role.rolname
            """,
            (runtime_principals, role),
        )
        return tuple(str(row[0]) for row in cursor.fetchall())


def _runtime_identity_root_set_role_denials(role: str) -> tuple[str, ...]:
    """Attempt SET ROLE from every topology pool and retain only 42501 denials."""
    denied_principals: list[str] = []
    for binding_ref in sorted(_DATABASE.bindings):
        runtime = _connect(_pool_dsn(binding_ref))
        current_user: str | None = None
        try:
            try:
                with runtime.cursor() as cursor:
                    cursor.execute("SELECT current_user")
                    current_user = str(cursor.fetchone()[0])
                    cursor.execute(
                        pg_sql.SQL("SET LOCAL ROLE {}").format(pg_sql.Identifier(role))
                    )
            except psycopg2.Error as exc:
                runtime.rollback()
                if str(exc.pgcode) != "42501":
                    raise AssertionError(
                        f"pool {binding_ref!r} SET ROLE proof failed with unexpected "
                        f"SQLSTATE {exc.pgcode!r}"
                    ) from exc
                if current_user is None:
                    raise AssertionError(
                        f"pool {binding_ref!r} did not expose current_user"
                    ) from exc
                denied_principals.append(current_user)
            else:
                runtime.rollback()
        finally:
            runtime.close()
    return tuple(sorted(denied_principals))


def _identity_root_control_state(
    connection: psycopg2.extensions.connection,
    classification: ModelRelationEvidence,
) -> ModelApplicationDatabaseIdentityRootControlState | None:
    if classification.identity_root_contract is None:
        return None
    role = classification.identity_root_control_role
    if role is None or classification.schema is None:
        raise AssertionError("identity-root classification lacks control authority")
    qualified_relation = f"{classification.schema}.{classification.name}"
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT role.rolcanlogin, role.rolsuper, role.rolbypassrls,
                   has_table_privilege(role.rolname, %s, 'INSERT'),
                   has_table_privilege(role.rolname, %s, 'SELECT')
            FROM pg_roles role
            WHERE role.rolname = %s
            """,
            (qualified_relation, qualified_relation, role),
        )
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise AssertionError(
            f"identity-root control role {role!r} must exist exactly once"
        )
    can_login, superuser, bypass_rls, can_insert, can_select = rows[0]
    observed: list[EnumApplicationDatabaseIdentityRootOperation] = []
    proof_ids: list[str] = []
    if can_insert and _prove_identity_root_creation(
        schema=classification.schema,
        name=classification.name,
        role=role,
    ):
        observed.append(EnumApplicationDatabaseIdentityRootOperation.TENANT_CREATION)
        proof_ids.append("postgres16:identity-root-control-create-and-runtime-deny")
    if can_select and _prove_identity_root_enumeration(
        schema=classification.schema,
        name=classification.name,
        role=role,
    ):
        observed.append(
            EnumApplicationDatabaseIdentityRootOperation.CROSS_TENANT_ENUMERATION
        )
        proof_ids.append("postgres16:identity-root-control-enumerate-and-runtime-scope")
    return ModelApplicationDatabaseIdentityRootControlState(
        role=role,
        role_can_login=can_login,
        role_superuser=superuser,
        role_bypass_rls=bypass_rls,
        runtime_membership_principals=(
            _runtime_identity_root_membership_principals(connection, role)
        ),
        runtime_set_role_denied_principals=(
            _runtime_identity_root_set_role_denials(role)
        ),
        declared_operations=classification.identity_root_control_operations,
        observed_operations=tuple(observed),
        behavioral_proof_ids=tuple(proof_ids),
    )


def _catalog_routine_dependency_states(
    cursor: psycopg2.extensions.cursor,
    target_relation_oid: int,
) -> tuple[ModelApplicationDatabaseRoutineDependencyState, ...]:
    """Load immutable routine definitions and exact catalog dependency edges."""
    cursor.execute(
        """
        SELECT routine.oid,
               namespace.nspname,
               routine.proname,
               language.lanname,
               routine.prosrc,
               routine.proargtypes::oid[],
               ARRAY(
                 SELECT NULLIF(argument.argument_name, '')
                 FROM unnest(
                   COALESCE(
                     routine.proargnames,
                     array_fill(
                       NULL::text,
                       ARRAY[
                         cardinality(
                           COALESCE(
                             routine.proallargtypes,
                             routine.proargtypes::oid[]
                           )
                         )
                       ]
                     )
                   ),
                   COALESCE(
                     routine.proargmodes,
                     array_fill('i'::"char", ARRAY[routine.pronargs])
                   )
                 ) AS argument(argument_name, argument_mode)
                 WHERE argument.argument_mode IN (
                   'i'::"char",
                   'b'::"char",
                   'v'::"char"
                 )
               ),
               routine.prorettype IN (
                 'pg_catalog.trigger'::regtype,
                 'pg_catalog.event_trigger'::regtype
               ),
               ARRAY(
                 SELECT DISTINCT dependency.refobjid
                 FROM pg_depend dependency
                 WHERE dependency.classid = 'pg_proc'::regclass
                   AND dependency.objid = routine.oid
                   AND dependency.refclassid = 'pg_proc'::regclass
                 ORDER BY dependency.refobjid
               ),
               ARRAY(
                 SELECT DISTINCT attribute.attname
                 FROM pg_depend dependency
                 JOIN pg_attribute attribute
                   ON attribute.attrelid = dependency.refobjid
                  AND attribute.attnum = dependency.refobjsubid
                 WHERE dependency.classid = 'pg_proc'::regclass
                   AND dependency.objid = routine.oid
                   AND dependency.refclassid = 'pg_class'::regclass
                   AND dependency.refobjid = %s
                   AND dependency.refobjsubid > 0
                 ORDER BY attribute.attname
               ),
               EXISTS (
                 SELECT 1
                 FROM pg_depend dependency
                 WHERE dependency.classid = 'pg_proc'::regclass
                   AND dependency.objid = routine.oid
                   AND dependency.refclassid = 'pg_class'::regclass
                   AND dependency.refobjid = %s
                   AND dependency.refobjsubid = 0
               )
        FROM pg_proc routine
        JOIN pg_namespace namespace ON namespace.oid = routine.pronamespace
        JOIN pg_language language ON language.oid = routine.prolang
        ORDER BY routine.oid
        """,
        (target_relation_oid, target_relation_oid),
    )
    return tuple(
        ModelApplicationDatabaseRoutineDependencyState(
            object_id=row[0],
            namespace=row[1],
            name=row[2],
            language=row[3],
            source_body=row[4],
            argument_type_ids=tuple(row[5]),
            argument_names=tuple(row[6]),
            returns_trigger=row[7],
            referenced_routine_ids=tuple(row[8]),
            referenced_target_columns=tuple(row[9]),
            references_target_whole_row=row[10],
        )
        for row in cursor.fetchall()
    )


def _catalog_surface_authority_columns(
    cursor: psycopg2.extensions.cursor,
    *,
    roots: Sequence[tuple[str, int]],
    target_relation_oid: int,
    target_composite_type_id: int,
    columns: Sequence[ModelApplicationDatabaseColumnState],
    routines: Sequence[ModelApplicationDatabaseRoutineDependencyState],
) -> tuple[str, ...]:
    """Resolve a surface through pg_depend and reachable routine definitions."""
    if not roots:
        return ()
    roots_by_class: dict[str, list[int]] = {}
    for class_name, object_id in roots:
        roots_by_class.setdefault(class_name, []).append(object_id)

    direct_columns: set[str] = set()
    direct_whole_row_reference = False
    routine_ids: set[int] = set()
    for class_name, object_ids in roots_by_class.items():
        cursor.execute(
            """
            WITH RECURSIVE dependency_walk(
              refclassid,
              refobjid,
              refobjsubid,
              visited
            ) AS (
              SELECT dependency.refclassid,
                     dependency.refobjid,
                     dependency.refobjsubid,
                     ARRAY[
                       dependency.refclassid::text || ':' ||
                       dependency.refobjid::text || ':' ||
                       dependency.refobjsubid::text
                     ]
              FROM pg_depend dependency
              WHERE dependency.classid = %s::regclass
                AND dependency.objid = ANY(%s::oid[])
              UNION ALL
              SELECT dependency.refclassid,
                     dependency.refobjid,
                     dependency.refobjsubid,
                     dependency_walk.visited || (
                       dependency.refclassid::text || ':' ||
                       dependency.refobjid::text || ':' ||
                       dependency.refobjsubid::text
                     )
              FROM dependency_walk
              JOIN pg_depend dependency
                ON dependency.classid = dependency_walk.refclassid
               AND dependency.objid = dependency_walk.refobjid
              WHERE dependency_walk.refclassid <> 'pg_class'::regclass
                AND NOT (
                  dependency.refclassid::text || ':' ||
                  dependency.refobjid::text || ':' ||
                  dependency.refobjsubid::text
                ) = ANY(dependency_walk.visited)
            )
            SELECT DISTINCT dependency_walk.refclassid = 'pg_proc'::regclass,
                            dependency_walk.refobjid,
                            attribute.attname,
                            dependency_walk.refclassid = 'pg_class'::regclass
                              AND dependency_walk.refobjid = %s
                              AND dependency_walk.refobjsubid = 0
                              AS references_target_whole_row
            FROM dependency_walk
            LEFT JOIN pg_attribute attribute
              ON dependency_walk.refclassid = 'pg_class'::regclass
             AND dependency_walk.refobjid = attribute.attrelid
             AND dependency_walk.refobjsubid = attribute.attnum
             AND dependency_walk.refobjid = %s
             AND dependency_walk.refobjsubid > 0
            WHERE dependency_walk.refclassid = 'pg_proc'::regclass
               OR attribute.attname IS NOT NULL
               OR (
                 dependency_walk.refclassid = 'pg_class'::regclass
                 AND dependency_walk.refobjid = %s
                 AND dependency_walk.refobjsubid = 0
               )
            """,
            (
                class_name,
                list(dict.fromkeys(object_ids)),
                target_relation_oid,
                target_relation_oid,
                target_relation_oid,
            ),
        )
        for (
            is_routine,
            object_id,
            column_name,
            references_target_whole_row,
        ) in cursor.fetchall():
            if is_routine:
                routine_ids.add(int(object_id))
            elif references_target_whole_row:
                direct_whole_row_reference = True
            elif column_name is not None:
                direct_columns.add(str(column_name))

    return resolve_application_database_authority_columns(
        target_columns=tuple(column.name for column in columns),
        target_composite_type_id=target_composite_type_id,
        direct_referenced_columns=tuple(direct_columns),
        direct_whole_row_reference=direct_whole_row_reference,
        root_routine_ids=tuple(routine_ids),
        routines=routines,
        governed_schemas=tuple(_DATABASE.schemas),
    )


def _table_state(
    connection: psycopg2.extensions.connection,
    declaration: ModelApplicationRelationDeclaration,
) -> ModelApplicationDatabaseRelationState:
    schema = declaration.schema
    name = declaration.name
    classification = _relation_evidence(declaration)
    columns = _column_states(connection, schema, name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT relation.oid, relation.reltype
            FROM pg_class relation
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s AND relation.relname = %s
            """,
            (schema, name),
        )
        relation_rows = cursor.fetchall()
        if len(relation_rows) != 1:
            raise AssertionError(
                f"{declaration.identity!r} requires exactly one catalog relation; "
                f"observed {len(relation_rows)}"
            )
        target_relation_oid, target_composite_type_id = relation_rows[0]
        cursor.execute(
            """
            SELECT attribute.attname
            FROM pg_index index_row
            CROSS JOIN LATERAL unnest(index_row.indkey)
                WITH ORDINALITY AS key_column(attnum, position)
            JOIN pg_attribute attribute
              ON attribute.attrelid = index_row.indrelid
             AND attribute.attnum = key_column.attnum
            JOIN pg_class relation ON relation.oid = index_row.indrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE index_row.indisprimary
              AND key_column.position <= index_row.indnkeyatts
              AND namespace.nspname = %s
              AND relation.relname = %s
            ORDER BY key_column.position
            """,
            (schema, name),
        )
        primary_key_columns = tuple(row[0] for row in cursor.fetchall())
        cursor.execute(
            """
            SELECT index_row.indexrelid,
                   index_relation.relname,
                   ARRAY(
                     SELECT attribute.attname
                     FROM unnest(index_row.indkey)
                         WITH ORDINALITY AS key_column(attnum, position)
                     JOIN pg_attribute attribute
                       ON attribute.attrelid = index_row.indrelid
                      AND attribute.attnum = key_column.attnum
                     WHERE key_column.position <= index_row.indnkeyatts
                     ORDER BY key_column.position
                   ),
                   pg_get_expr(index_row.indexprs, index_row.indrelid),
                   pg_get_expr(index_row.indpred, index_row.indrelid)
            FROM pg_index index_row
            JOIN pg_class relation ON relation.oid = index_row.indrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            JOIN pg_class index_relation ON index_relation.oid = index_row.indexrelid
            WHERE index_row.indisunique
              AND namespace.nspname = %s
              AND relation.relname = %s
            ORDER BY index_relation.relname
            """,
            (schema, name),
        )
        unique_index_rows = tuple(
            (row[0], tuple(row[2]), row[3], row[4]) for row in cursor.fetchall()
        )
        cursor.execute(
            """
            SELECT constraint_row.conname,
                   array_agg(attribute.attname ORDER BY key_column.position)
            FROM pg_constraint constraint_row
            JOIN pg_class relation ON relation.oid = constraint_row.conrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL unnest(constraint_row.conkey)
                WITH ORDINALITY AS key_column(attnum, position)
            JOIN pg_attribute attribute
              ON attribute.attrelid = constraint_row.conrelid
             AND attribute.attnum = key_column.attnum
            WHERE constraint_row.contype = 'f'
              AND namespace.nspname = %s
              AND relation.relname = %s
            GROUP BY constraint_row.conname
            ORDER BY constraint_row.conname
            """,
            (schema, name),
        )
        foreign_key_column_sets = tuple(tuple(row[1]) for row in cursor.fetchall())
        cursor.execute(
            """
            SELECT attribute.attname
            FROM pg_partitioned_table partitioned
            JOIN pg_class relation ON relation.oid = partitioned.partrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL unnest(partitioned.partattrs)
                WITH ORDINALITY AS key_column(attnum, position)
            JOIN pg_attribute attribute
              ON attribute.attrelid = partitioned.partrelid
             AND attribute.attnum = key_column.attnum
            WHERE namespace.nspname = %s
              AND relation.relname = %s
            ORDER BY key_column.position
            """,
            (schema, name),
        )
        direct_partition_key_columns = tuple(row[0] for row in cursor.fetchall())
        cursor.execute(
            """
            SELECT pg_get_expr(partitioned.partexprs, partitioned.partrelid)
            FROM pg_partitioned_table partitioned
            JOIN pg_class relation ON relation.oid = partitioned.partrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s
              AND relation.relname = %s
            """,
            (schema, name),
        )
        partition_expressions = tuple(
            row[0] for row in cursor.fetchall() if row[0] is not None
        )
        cursor.execute(
            """
            SELECT relation.relrowsecurity, relation.relforcerowsecurity
            FROM pg_class relation
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s AND relation.relname = %s
            """,
            (schema, name),
        )
        rls_enabled, rls_forced = cursor.fetchone()
        cursor.execute(
            """
            SELECT policy.polname, policy.polpermissive, policy.polcmd,
                   ARRAY(
                     SELECT CASE
                              WHEN policy_role.role_oid = 0 THEN 'PUBLIC'
                              ELSE role.rolname
                            END
                     FROM unnest(policy.polroles) WITH ORDINALITY
                       AS policy_role(role_oid, position)
                     LEFT JOIN pg_roles role ON role.oid = policy_role.role_oid
                     ORDER BY policy_role.position
                   ),
                   pg_get_expr(policy.polqual, policy.polrelid),
                   pg_get_expr(policy.polwithcheck, policy.polrelid)
            FROM pg_policy policy
            JOIN pg_class relation ON relation.oid = policy.polrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s AND relation.relname = %s
            ORDER BY policy.polname
            """,
            (schema, name),
        )
        policies = tuple(
            ModelApplicationDatabasePolicyState(
                name=row[0],
                permissive=row[1],
                command=_POLICY_COMMANDS[row[2]],
                roles=tuple(row[3]),
                using_expression=row[4],
                with_check_expression=row[5],
            )
            for row in cursor.fetchall()
        )
        cursor.execute(
            """
            SELECT policy.oid
            FROM pg_policy policy
            JOIN pg_class relation ON relation.oid = policy.polrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s AND relation.relname = %s
            ORDER BY policy.oid
            """,
            (schema, name),
        )
        policy_roots = tuple(("pg_policy", row[0]) for row in cursor.fetchall())
        cursor.execute(
            """
            SELECT CASE constraint_row.contype
                     WHEN 'c' THEN pg_get_expr(
                       constraint_row.conbin,
                       constraint_row.conrelid
                     )
                     ELSE pg_get_constraintdef(constraint_row.oid, true)
                   END
            FROM pg_constraint constraint_row
            JOIN pg_class relation ON relation.oid = constraint_row.conrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE constraint_row.contype IN ('c', 'x')
              AND namespace.nspname = %s
              AND relation.relname = %s
            UNION ALL
            SELECT pg_get_triggerdef(trigger_row.oid, true)
            FROM pg_trigger trigger_row
            JOIN pg_class relation ON relation.oid = trigger_row.tgrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE NOT trigger_row.tgisinternal
              AND namespace.nspname = %s
              AND relation.relname = %s
            UNION ALL
            SELECT pg_get_functiondef(trigger_row.tgfoid)
            FROM pg_trigger trigger_row
            JOIN pg_class relation ON relation.oid = trigger_row.tgrelid
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE NOT trigger_row.tgisinternal
              AND namespace.nspname = %s
              AND relation.relname = %s
            """,
            (schema, name, schema, name, schema, name),
        )
        write_eligibility_expressions = tuple(
            row[0] for row in cursor.fetchall() if row[0] is not None
        )
        cursor.execute(
            """
            WITH RECURSIVE dependent_views(rewrite_oid, relation_oid) AS (
              SELECT DISTINCT rewrite_row.oid, dependent_relation.oid
              FROM pg_depend dependency
              JOIN pg_rewrite rewrite_row ON rewrite_row.oid = dependency.objid
              JOIN pg_class dependent_relation
                ON dependent_relation.oid = rewrite_row.ev_class
              WHERE dependency.classid = 'pg_rewrite'::regclass
                AND dependency.refclassid = 'pg_class'::regclass
                AND dependency.refobjid = %s
                AND dependent_relation.relkind IN ('v', 'm')
                AND dependent_relation.oid <> %s
              UNION
              SELECT DISTINCT rewrite_row.oid, dependent_relation.oid
              FROM dependent_views source_view
              JOIN pg_depend dependency
                ON dependency.refclassid = 'pg_class'::regclass
               AND dependency.refobjid = source_view.relation_oid
              JOIN pg_rewrite rewrite_row ON rewrite_row.oid = dependency.objid
              JOIN pg_class dependent_relation
                ON dependent_relation.oid = rewrite_row.ev_class
              WHERE dependency.classid = 'pg_rewrite'::regclass
                AND dependent_relation.relkind IN ('v', 'm')
                AND dependent_relation.oid <> source_view.relation_oid
            )
            SELECT rewrite_oid, pg_get_viewdef(relation_oid, true)
            FROM dependent_views
            ORDER BY relation_oid
            """,
            (target_relation_oid, target_relation_oid),
        )
        dependent_view_rows = tuple(cursor.fetchall())
        dependent_view_expressions = tuple(row[1] for row in dependent_view_rows)
        cursor.execute(
            """
            SELECT 'pg_constraint', constraint_row.oid
            FROM pg_constraint constraint_row
            WHERE constraint_row.conrelid = %s
              AND constraint_row.contype IN ('c', 'x')
            UNION ALL
            SELECT 'pg_trigger', trigger_row.oid
            FROM pg_trigger trigger_row
            WHERE trigger_row.tgrelid = %s
              AND NOT trigger_row.tgisinternal
            """,
            (target_relation_oid, target_relation_oid),
        )
        write_eligibility_roots = tuple(
            (str(row[0]), int(row[1])) for row in cursor.fetchall()
        )
        cursor.execute(
            """
            SELECT attribute.attname, default_row.oid
            FROM pg_attribute attribute
            JOIN pg_attrdef default_row
              ON default_row.adrelid = attribute.attrelid
             AND default_row.adnum = attribute.attnum
            WHERE attribute.attrelid = %s
              AND attribute.attgenerated <> ''
            ORDER BY attribute.attnum
            """,
            (target_relation_oid,),
        )
        generated_expression_roots = tuple(cursor.fetchall())
    has_source_tenant = any(column.name == "source_tenant_id" for column in columns)
    if has_source_tenant != (
        classification.source_tenant_provenance_contract is not None
    ):
        raise AssertionError(
            f"{declaration.identity!r} source_tenant_id and its typed provenance "
            "classification must be declared together"
        )
    semantic_fields = {
        "deduplication_key_columns": classification.deduplication_key_columns,
        "authorization_dependency_columns": (
            classification.authorization_dependency_columns
        ),
        "write_eligibility_dependency_columns": (
            classification.write_eligibility_dependency_columns
        ),
    }
    missing_semantics = [
        name for name, value in semantic_fields.items() if value is None
    ]
    if missing_semantics:
        raise AssertionError(
            f"{declaration.identity!r} lacks explicit typed semantic fields: "
            f"{missing_semantics!r}"
        )
    policy_expressions = tuple(
        expression
        for policy in policies
        for expression in (policy.using_expression, policy.with_check_expression)
        if expression is not None
    )

    def referenced_columns(expressions: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            column.name
            for column in columns
            if any(
                re.search(
                    rf"(?<![a-zA-Z0-9_]){re.escape(column.name)}(?![a-zA-Z0-9_])",
                    expression,
                )
                for expression in expressions
            )
        )

    catalog_unique_dependencies: dict[int, tuple[str, ...]] = {}
    catalog_generated_dependencies: dict[str, tuple[str, ...]] = {}
    catalog_partition_dependencies: tuple[str, ...] = ()
    catalog_authorization_dependencies: tuple[str, ...] = ()
    catalog_write_dependencies: tuple[str, ...] = ()
    if has_source_tenant:
        with connection.cursor() as cursor:
            routine_dependencies = _catalog_routine_dependency_states(
                cursor,
                target_relation_oid,
            )
            for index_oid, *_ in unique_index_rows:
                catalog_unique_dependencies[index_oid] = (
                    _catalog_surface_authority_columns(
                        cursor,
                        roots=(("pg_class", index_oid),),
                        target_relation_oid=target_relation_oid,
                        target_composite_type_id=target_composite_type_id,
                        columns=columns,
                        routines=routine_dependencies,
                    )
                )
            for column_name, attribute_default_oid in generated_expression_roots:
                catalog_generated_dependencies[column_name] = (
                    _catalog_surface_authority_columns(
                        cursor,
                        roots=(("pg_attrdef", attribute_default_oid),),
                        target_relation_oid=target_relation_oid,
                        target_composite_type_id=target_composite_type_id,
                        columns=columns,
                        routines=routine_dependencies,
                    )
                )
            if direct_partition_key_columns or partition_expressions:
                catalog_partition_dependencies = _catalog_surface_authority_columns(
                    cursor,
                    roots=(("pg_class", target_relation_oid),),
                    target_relation_oid=target_relation_oid,
                    target_composite_type_id=target_composite_type_id,
                    columns=columns,
                    routines=routine_dependencies,
                )
            catalog_authorization_dependencies = _catalog_surface_authority_columns(
                cursor,
                roots=(
                    *policy_roots,
                    *(("pg_rewrite", row[0]) for row in dependent_view_rows),
                ),
                target_relation_oid=target_relation_oid,
                target_composite_type_id=target_composite_type_id,
                columns=columns,
                routines=routine_dependencies,
            )
            catalog_write_dependencies = _catalog_surface_authority_columns(
                cursor,
                roots=write_eligibility_roots,
                target_relation_oid=target_relation_oid,
                target_composite_type_id=target_composite_type_id,
                columns=columns,
                routines=routine_dependencies,
            )

    generated_dependencies = {
        column.name: tuple(
            dict.fromkeys(
                (
                    *referenced_columns((column.generated_expression,)),
                    *catalog_generated_dependencies.get(column.name, ()),
                )
            )
        )
        for column in columns
        if column.generated_expression is not None
    }

    def expand_generated_dependencies(column_names: Sequence[str]) -> tuple[str, ...]:
        """Make semantic authority through generated aliases explicit."""
        expanded: list[str] = []
        pending = list(column_names)
        while pending:
            column_name = pending.pop(0)
            if column_name in expanded:
                continue
            expanded.append(column_name)
            pending.extend(generated_dependencies.get(column_name, ()))
        return tuple(expanded)

    unique_index_column_sets = tuple(
        expand_generated_dependencies(combined)
        for index_oid, direct_columns, expression, predicate in unique_index_rows
        if (
            combined := tuple(
                dict.fromkeys(
                    (
                        *direct_columns,
                        *referenced_columns((expression,) if expression else ()),
                        *referenced_columns((predicate,) if predicate else ()),
                        *catalog_unique_dependencies.get(index_oid, ()),
                    )
                )
            )
        )
    )
    partition_key_columns = expand_generated_dependencies(
        tuple(
            dict.fromkeys(
                (
                    *direct_partition_key_columns,
                    *referenced_columns(partition_expressions),
                    *catalog_partition_dependencies,
                )
            )
        )
    )
    deduplication_columns = classification.deduplication_key_columns
    authorization_columns = classification.authorization_dependency_columns
    write_eligibility_columns = classification.write_eligibility_dependency_columns
    assert deduplication_columns is not None
    assert authorization_columns is not None
    assert write_eligibility_columns is not None
    return ModelApplicationDatabaseRelationState(
        declaration=declaration,
        columns=columns,
        primary_key_columns=expand_generated_dependencies(primary_key_columns),
        unique_index_column_sets=unique_index_column_sets,
        foreign_key_column_sets=tuple(
            expand_generated_dependencies(column_set)
            for column_set in foreign_key_column_sets
        ),
        partition_key_columns=partition_key_columns,
        deduplication_key_columns=expand_generated_dependencies(deduplication_columns),
        authorization_dependency_columns=expand_generated_dependencies(
            tuple(
                dict.fromkeys(
                    (
                        *authorization_columns,
                        *referenced_columns(
                            (*policy_expressions, *dependent_view_expressions)
                        ),
                        *catalog_authorization_dependencies,
                    )
                )
            )
        ),
        write_eligibility_dependency_columns=expand_generated_dependencies(
            tuple(
                dict.fromkeys(
                    (
                        *write_eligibility_columns,
                        *referenced_columns(write_eligibility_expressions),
                        *catalog_write_dependencies,
                    )
                )
            )
        ),
        rls_enabled=rls_enabled,
        rls_forced=rls_forced,
        policies=policies,
        tenant_identity_column=classification.tenant_identity_column,
        identity_root_contract=classification.identity_root_contract,
        identity_root_control_state=_identity_root_control_state(
            connection,
            classification,
        ),
        canonical_policy_name=classification.canonical_policy_name,
        source_tenant_provenance_contract=(
            classification.source_tenant_provenance_contract
        ),
    )


def _surface_count(query: pg_sql.Composable, tenant_id: UUID | None) -> int:
    connection = _connect(_pool_dsn("app_dashboard"))
    try:
        connection.autocommit = False
        with connection.cursor() as cursor:
            if tenant_id is not None:
                cursor.execute("SET LOCAL app.tenant_id = %s", (str(tenant_id),))
            cursor.execute(query)
            count = int(cursor.fetchone()[0])
        connection.rollback()
        return count
    finally:
        connection.close()


def _malformed_context_denied(query: pg_sql.Composable) -> bool:
    connection = _connect(_pool_dsn("app_dashboard"))
    try:
        connection.autocommit = False
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL app.tenant_id = 'not-a-uuid'")
                cursor.execute(query)
        except psycopg2.Error as exc:
            connection.rollback()
            return str(exc.pgcode) == "22P02"
        connection.rollback()
        return False
    finally:
        connection.close()


def _behavioral_evidence(
    query: pg_sql.Composable,
) -> ModelApplicationDatabaseTenantIsolationEvidence:
    observed = {
        tenant_id: _surface_count(query, tenant_id)
        for tenant_id in _EXPECTED_TENANT_ROWS
    }
    return ModelApplicationDatabaseTenantIsolationEvidence(
        expected_rows_by_tenant=_EXPECTED_TENANT_ROWS,
        observed_rows_by_tenant=observed,
        unset_context_rows=_surface_count(query, None),
        malformed_context_denied=_malformed_context_denied(query),
    )


def _view_state(
    connection: psycopg2.extensions.connection,
    declaration: ModelApplicationRelationDeclaration,
    evidence: ModelApplicationDatabaseTenantIsolationEvidence | None,
) -> ModelApplicationDatabaseRelationState:
    _relation_evidence(declaration)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COALESCE(relation.reloptions, ARRAY[]::text[])
            FROM pg_class relation
            JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = %s AND relation.relname = %s
            """,
            (declaration.schema, declaration.name),
        )
        options = set(cursor.fetchone()[0])
    return ModelApplicationDatabaseRelationState(
        declaration=declaration,
        columns=_column_states(connection, declaration.schema, declaration.name),
        security_invoker="security_invoker=true" in options,
        view_tenant_isolation_evidence=evidence,
    )


def _function_state(
    connection: psycopg2.extensions.connection,
    declaration: ModelApplicationRelationDeclaration,
    evidence: ModelApplicationDatabaseTenantIsolationEvidence | None,
) -> ModelApplicationDatabaseRelationState:
    authority = _database_object_evidence(declaration)
    signature = declaration.function_signature
    if signature is None:
        raise AssertionError(
            f"{declaration.identity!r} lacks an exact routine signature"
        )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT owner.rolname, routine.prosecdef,
                   COALESCE(routine.proconfig, ARRAY[]::text[]),
                   EXISTS (
                     SELECT 1
                     FROM aclexplode(
                       COALESCE(routine.proacl, acldefault('f', routine.proowner))
                     ) acl
                     WHERE acl.grantee = 0 AND acl.privilege_type = 'EXECUTE'
                   ) AS public_execute,
                   language.lanname,
                   routine.prosrc,
                   routine.prosqlbody::text,
                   routine.proleakproof,
                   routine.provolatile,
                   routine.proparallel,
                   routine.prokind,
                   routine.proisstrict,
                   routine.proretset,
                   pg_get_function_result(routine.oid)
            FROM pg_proc routine
            JOIN pg_namespace namespace ON namespace.oid = routine.pronamespace
            JOIN pg_roles owner ON owner.oid = routine.proowner
            JOIN pg_language language ON language.oid = routine.prolang
            WHERE namespace.nspname = %s
              AND routine.proname = %s
              AND '(' || pg_get_function_identity_arguments(routine.oid) || ')' = %s
            """,
            (declaration.schema, declaration.name, signature),
        )
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise AssertionError(
            f"{declaration.identity!r} requires exactly one live routine; "
            f"observed {len(rows)}"
        )
    (
        owner,
        security_definer,
        config,
        public_execute,
        language,
        source_body,
        parsed_sql_body,
        leakproof,
        volatility,
        parallel,
        kind,
        strict,
        returns_set,
        result_type,
    ) = rows[0]
    search_path: tuple[str, ...] = ()
    for setting in config:
        if setting.startswith("search_path="):
            search_path = tuple(
                part.strip() for part in setting.removeprefix("search_path=").split(",")
            )
    definition_sha256 = application_database_function_definition_sha256(
        schema=declaration.schema,
        name=declaration.name,
        signature=signature,
        language=language,
        source_body=source_body,
        parsed_sql_body=parsed_sql_body,
        security_definer=security_definer,
        leakproof=leakproof,
        volatility=volatility,
        parallel=parallel,
        config=config,
        kind=kind,
        strict=strict,
        returns_set=returns_set,
        result_type=result_type,
    )
    return ModelApplicationDatabaseRelationState(
        declaration=declaration,
        function_state=ModelApplicationDatabaseFunctionState(
            owner=owner,
            security_definer=security_definer,
            search_path=search_path,
            public_execute=public_execute,
            audit_id=authority.audit_id,
            definition_sha256=definition_sha256,
            audited_definition_sha256=authority.definition_sha256,
            tenant_isolation_evidence=evidence,
        ),
    )


def _catalog_identities(
    connection: psycopg2.extensions.connection,
) -> tuple[ModelApplicationDatabaseCatalogIdentity, ...]:
    """Read every class, routine, type, and extension in application schemas."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT schema_name, object_name, object_kind, function_signature
            FROM (
                SELECT namespace.nspname AS schema_name,
                       relation.relname AS object_name,
                       CASE relation.relkind
                         WHEN 'r' THEN 'table'
                         WHEN 'p' THEN 'table'
                         WHEN 'v' THEN 'view'
                         WHEN 'm' THEN 'materialized_view'
                         WHEN 'S' THEN 'sequence'
                         WHEN 'f' THEN 'foreign_table'
                       END AS object_kind,
                       NULL::text AS function_signature
                FROM pg_class relation
                JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
                WHERE relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
                  AND namespace.nspname !~ '^pg_'
                  AND namespace.nspname <> 'information_schema'

                UNION ALL

                SELECT namespace.nspname,
                       routine.proname,
                       CASE routine.prokind
                         WHEN 'p' THEN 'procedure'
                         WHEN 'a' THEN 'aggregate'
                         WHEN 'w' THEN 'window_function'
                         ELSE 'function'
                       END,
                       '(' || pg_get_function_identity_arguments(routine.oid) || ')'
                FROM pg_proc routine
                JOIN pg_namespace namespace ON namespace.oid = routine.pronamespace
                WHERE routine.prokind IN ('f', 'p', 'a', 'w')
                  AND namespace.nspname !~ '^pg_'
                  AND namespace.nspname <> 'information_schema'

                UNION ALL

                SELECT namespace.nspname,
                       type_row.typname,
                       CASE type_row.typtype
                         WHEN 'b' THEN 'base_type'
                         WHEN 'r' THEN 'range_type'
                         WHEN 'm' THEN 'multirange_type'
                         ELSE 'type'
                       END,
                       NULL::text
                FROM pg_type type_row
                JOIN pg_namespace namespace ON namespace.oid = type_row.typnamespace
                LEFT JOIN pg_class relation ON relation.oid = type_row.typrelid
                WHERE (
                    type_row.typtype IN ('d', 'e', 'r', 'm')
                    OR (type_row.typtype = 'c' AND relation.relkind = 'c')
                    OR (
                        type_row.typtype = 'b'
                        AND type_row.typelem = 0
                        AND type_row.typrelid = 0
                    )
                )
                  AND namespace.nspname !~ '^pg_'
                  AND namespace.nspname <> 'information_schema'

                UNION ALL

                SELECT namespace.nspname,
                       extension.extname,
                       'extension',
                       NULL::text
                FROM pg_extension extension
                JOIN pg_namespace namespace ON namespace.oid = extension.extnamespace
                WHERE namespace.nspname !~ '^pg_'
                  AND namespace.nspname <> 'information_schema'
            ) catalog
            ORDER BY schema_name, object_name, object_kind, function_signature
            """
        )
        return tuple(
            ModelApplicationDatabaseCatalogIdentity(
                schema=row[0],
                name=row[1],
                kind=EnumApplicationInventoryObjectKind(row[2]),
                function_signature=row[3],
            )
            for row in cursor.fetchall()
        )


def _ownership_inventory(
    observed: Sequence[ModelApplicationDatabaseCatalogIdentity],
) -> ModelApplicationRelationInventory:
    """Project the exhaustive census into the predecessor's ownership validator."""
    live_relations: list[ModelLiveApplicationRelation] = []
    excluded_objects: list[str] = []
    for identity in observed:
        relation_kind = _CATALOG_TO_RELATION_KIND.get(identity.kind)
        if relation_kind is None:
            excluded_objects.append(
                f"{identity.schema}.{identity.name}:{identity.kind.value}"
            )
            continue
        live_relations.append(
            ModelLiveApplicationRelation(
                name=identity.name,
                database_ref="application",
                schema=identity.schema,
                kind=relation_kind,
                domain=_TOPOLOGY.schema_domain("application", identity.schema),
                function_signature=identity.function_signature,
            )
        )
    return ModelApplicationRelationInventory(
        schema_version="1.0",
        relations=tuple(live_relations),
        completion_status="complete",
        source_relation_count=len(observed),
        excluded_database_objects=tuple(excluded_objects),
    )


def _authoritative_declarations(
    observed: Sequence[ModelApplicationDatabaseCatalogIdentity],
) -> tuple[ModelApplicationRelationDeclaration, ...]:
    """Resolve exactly one owner per live relation from the checked manifest."""
    report = validate_application_relation_ownership(
        topology=_TOPOLOGY,
        node_contract_paths=(),
        service_manifest_paths=(_OWNERSHIP_MANIFEST,),
        inventory=_ownership_inventory(observed),
    )
    if not report.is_valid:
        messages = tuple(
            f"{violation.code.value}: {violation.message}"
            for violation in report.violations
        )
        raise AssertionError(f"authoritative ownership proof failed: {messages}")
    owners = tuple(
        declaration
        for declaration in report.declarations
        if declaration.owner_declaration is not None
    )
    owner_identities = [owner.identity for owner in owners]
    if len(owner_identities) != len(set(owner_identities)):
        raise AssertionError("ownership report did not resolve unique owner identities")
    if len(owners) != len(_ownership_inventory(observed).relations):
        raise AssertionError(
            "ownership report owner count does not match live relation projection"
        )
    return tuple(
        sorted(
            owners,
            key=lambda owner: tuple(
                "" if item is None else str(item) for item in owner.identity
            ),
        )
    )


def _tenant_surface_query(
    declaration: ModelApplicationRelationDeclaration,
) -> pg_sql.Composable:
    if declaration.kind is EnumApplicationRelationKind.VIEW:
        return pg_sql.SQL("SELECT count(*) FROM {}.{}").format(
            pg_sql.Identifier(declaration.schema),
            pg_sql.Identifier(declaration.name),
        )
    if declaration.kind is EnumApplicationRelationKind.FUNCTION:
        return pg_sql.SQL("SELECT {}.{}()").format(
            pg_sql.Identifier(declaration.schema),
            pg_sql.Identifier(declaration.name),
        )
    raise AssertionError(
        f"tenant behavioral surface is unsupported for {declaration.kind.value}"
    )


def _relation_state(
    connection: psycopg2.extensions.connection,
    declaration: ModelApplicationRelationDeclaration,
) -> ModelApplicationDatabaseRelationState:
    """Join one manifest-authoritative declaration to live catalog evidence."""
    if declaration.kind is EnumApplicationRelationKind.TABLE:
        return _table_state(connection, declaration)
    evidence = (
        _behavioral_evidence(_tenant_surface_query(declaration))
        if declaration.domain is EnumDatabaseSchemaDomain.TENANT
        and declaration.kind
        in {EnumApplicationRelationKind.VIEW, EnumApplicationRelationKind.FUNCTION}
        else None
    )
    if declaration.kind is EnumApplicationRelationKind.VIEW:
        return _view_state(connection, declaration, evidence)
    if declaration.kind is EnumApplicationRelationKind.FUNCTION:
        return _function_state(connection, declaration, evidence)
    if declaration.kind in {
        EnumApplicationRelationKind.MATERIALIZED_VIEW,
        EnumApplicationRelationKind.FOREIGN_TABLE,
    }:
        _relation_evidence(declaration)
        return ModelApplicationDatabaseRelationState(
            declaration=declaration,
            columns=_column_states(connection, declaration.schema, declaration.name),
        )
    raise AssertionError(f"unsupported application relation kind: {declaration.kind}")


def _pool_identities() -> tuple[ModelApplicationDatabasePoolIdentity, ...]:
    identities: list[ModelApplicationDatabasePoolIdentity] = []
    for binding_ref in _DATABASE.bindings:
        connection = _connect(_pool_dsn(binding_ref))
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), current_user")
                current_database, current_user = cursor.fetchone()
        finally:
            connection.close()
        identities.append(
            ModelApplicationDatabasePoolIdentity(
                pool=binding_ref,
                current_database=current_database,
                current_user=current_user,
            )
        )
    return tuple(identities)


def _assert_red(control: str, violations: Sequence[str], expected: str) -> None:
    text = "\n".join(violations)
    if expected not in text:
        raise AssertionError(
            f"seeded control {control!r} did not discriminate {expected!r}: {text}"
        )
    print(f"domain_control={control} status=PASS")


def _run_relation_controls(
    identity_root: ModelApplicationDatabaseRelationState,
    tenant: ModelApplicationDatabaseRelationState,
    internal: ModelApplicationDatabaseRelationState,
    view: ModelApplicationDatabaseRelationState,
    function: ModelApplicationDatabaseRelationState,
) -> int:
    tenant_column = next(
        column for column in tenant.columns if column.name == "tenant_id"
    )
    canonical = tenant.policies[0]
    observed_function = function.function_state
    if observed_function is None:
        raise AssertionError("green function state is missing")
    root_control = identity_root.identity_root_control_state
    if root_control is None:
        raise AssertionError("green identity-root control state is missing")
    controls: tuple[tuple[str, ModelApplicationDatabaseRelationState, str], ...] = (
        (
            "identity-root-runtime-login",
            identity_root.model_copy(
                update={
                    "identity_root_control_state": root_control.model_copy(
                        update={"role_can_login": True}
                    )
                }
            ),
            "NOLOGIN",
        ),
        (
            "identity-root-unproven-enumeration",
            identity_root.model_copy(
                update={
                    "identity_root_control_state": root_control.model_copy(
                        update={
                            "observed_operations": (
                                EnumApplicationDatabaseIdentityRootOperation.TENANT_CREATION,
                            ),
                            "behavioral_proof_ids": (
                                "postgres16:identity-root-control-create-and-runtime-deny",
                            ),
                        }
                    )
                }
            ),
            "differ from the declared operation set",
        ),
        (
            "tenant-text-key",
            tenant.model_copy(
                update={
                    "columns": tuple(
                        column.model_copy(update={"data_type": "text"})
                        if column.name == "tenant_id"
                        else column
                        for column in tenant.columns
                    )
                }
            ),
            "UUID",
        ),
        (
            "tenant-nullable",
            tenant.model_copy(
                update={
                    "columns": tuple(
                        tenant_column.model_copy(update={"nullable": True})
                        if column.name == "tenant_id"
                        else column
                        for column in tenant.columns
                    )
                }
            ),
            "NOT NULL",
        ),
        (
            "tenant-default",
            tenant.model_copy(
                update={
                    "columns": tuple(
                        tenant_column.model_copy(update={"default_expression": "0"})
                        if column.name == "tenant_id"
                        else column
                        for column in tenant.columns
                    )
                }
            ),
            "default",
        ),
        (
            "missing-enable-rls",
            tenant.model_copy(update={"rls_enabled": False}),
            "ENABLE ROW LEVEL SECURITY",
        ),
        (
            "missing-force-rls",
            tenant.model_copy(update={"rls_forced": False}),
            "FORCE ROW LEVEL SECURITY",
        ),
        (
            "using-drift",
            tenant.model_copy(
                update={
                    "policies": (
                        canonical.model_copy(update={"using_expression": "true"}),
                    )
                }
            ),
            "USING",
        ),
        (
            "with-check-drift",
            tenant.model_copy(
                update={
                    "policies": (
                        canonical.model_copy(update={"with_check_expression": "true"}),
                    )
                }
            ),
            "WITH CHECK",
        ),
        (
            "uncontracted-identity-root",
            tenant.model_copy(update={"tenant_identity_column": "event_id"}),
            "identity-root contract",
        ),
        (
            "owner-security-view",
            view.model_copy(update={"security_invoker": False}),
            "security_invoker",
        ),
        (
            "unproven-security-view",
            view.model_copy(update={"view_tenant_isolation_evidence": None}),
            "behavioral evidence",
        ),
        (
            "internal-tenant-id",
            internal.model_copy(
                update={
                    "columns": (
                        *internal.columns,
                        ModelApplicationDatabaseColumnState(
                            name="tenant_id",
                            data_type="uuid",
                            nullable=False,
                        ),
                    )
                }
            ),
            "tenant_id",
        ),
        (
            "uncontracted-source-tenant",
            internal.model_copy(update={"source_tenant_provenance_contract": None}),
            "provenance contract",
        ),
        (
            "source-tenant-uniqueness-authority",
            internal.model_copy(
                update={
                    "unique_index_column_sets": (
                        *internal.unique_index_column_sets,
                        ("source_tenant_id",),
                    )
                }
            ),
            "drive uniqueness",
        ),
        (
            "source-tenant-foreign-key-authority",
            internal.model_copy(
                update={"foreign_key_column_sets": (("source_tenant_id",),)}
            ),
            "drive foreign key",
        ),
        (
            "source-tenant-partition-authority",
            internal.model_copy(
                update={"partition_key_columns": ("source_tenant_id",)}
            ),
            "drive partition",
        ),
        (
            "source-tenant-deduplication-authority",
            internal.model_copy(
                update={
                    "deduplication_key_columns": (
                        *internal.deduplication_key_columns,
                        "source_tenant_id",
                    )
                }
            ),
            "drive deduplication",
        ),
        (
            "source-tenant-authorization-authority",
            internal.model_copy(
                update={"authorization_dependency_columns": ("source_tenant_id",)}
            ),
            "drive authorization",
        ),
        (
            "source-tenant-write-eligibility-authority",
            internal.model_copy(
                update={"write_eligibility_dependency_columns": ("source_tenant_id",)}
            ),
            "drive write eligibility",
        ),
        (
            "unsafe-security-definer",
            function.model_copy(
                update={
                    "function_state": observed_function.model_copy(
                        update={
                            "owner": "app_dashboard",
                            "search_path": ("public", "pg_temp"),
                            "public_execute": True,
                            "audit_id": None,
                        }
                    )
                }
            ),
            "PUBLIC EXECUTE",
        ),
        (
            "unproven-security-definer",
            function.model_copy(
                update={
                    "function_state": observed_function.model_copy(
                        update={"tenant_isolation_evidence": None}
                    )
                }
            ),
            "behavioral evidence",
        ),
        (
            "security-definer-definition-audit-drift",
            function.model_copy(
                update={
                    "function_state": observed_function.model_copy(
                        update={"definition_sha256": "0" * 64}
                    )
                }
            ),
            "does not match the audited definition hash",
        ),
    )
    for control, state, expected in controls:
        _assert_red(
            control,
            validate_application_database_relation_states((state,), _TOPOLOGY),
            expected,
        )
    return len(controls)


def _run_catalog_controls(
    admin: psycopg2.extensions.connection,
    states: tuple[ModelApplicationDatabaseRelationState, ...],
    observed: tuple[ModelApplicationDatabaseCatalogIdentity, ...],
    authority: tuple[ModelApplicationDatabaseCatalogIdentity, ...],
) -> int:
    _assert_red(
        "incomplete-catalog-census",
        validate_application_database_catalog_census(
            states,
            observed[1:],
            _TOPOLOGY,
            authoritative_identities=authority,
        ),
        "missing",
    )
    _assert_red(
        "empty-authoritative-relation-set",
        validate_application_database_catalog_census(
            (),
            observed,
            _TOPOLOGY,
            authoritative_identities=(),
        ),
        "cannot be empty",
    )
    admin.commit()
    with admin.cursor() as cursor:
        cursor.execute("CREATE UNLOGGED TABLE public.rogue_shadow (id uuid)")
    leaked_catalog = _catalog_identities(admin)
    _assert_red(
        "public-catalog-leak",
        validate_application_database_catalog_census(
            states,
            leaked_catalog,
            _TOPOLOGY,
            authoritative_identities=authority,
        ),
        "undeclared",
    )
    admin.rollback()
    return 3


def _run_live_source_tenant_controls(
    admin: psycopg2.extensions.connection,
    declaration: ModelApplicationRelationDeclaration,
) -> int:
    """Prove catalog extraction sees hidden source-tenant authority surfaces."""
    admin.commit()
    controls: tuple[tuple[str, str, str | tuple[str, ...]], ...] = (
        (
            "source-tenant-check-constraint",
            """
            ALTER TABLE omninode_internal.runtime_state
            ADD CONSTRAINT runtime_state_source_write_guard
            CHECK (source_tenant_id IS NULL)
            """,
            "drive write eligibility",
        ),
        (
            "source-tenant-partial-unique-predicate",
            """
            CREATE UNIQUE INDEX runtime_state_partial_source_guard
            ON omninode_internal.runtime_state (payload)
            WHERE source_tenant_id IS NOT NULL
            """,
            "drive uniqueness",
        ),
        (
            "source-tenant-generated-unique-alias",
            """
            ALTER TABLE omninode_internal.runtime_state
            ADD COLUMN source_tenant_copy uuid
            GENERATED ALWAYS AS (source_tenant_id) STORED;
            CREATE UNIQUE INDEX runtime_state_generated_source_guard
            ON omninode_internal.runtime_state (source_tenant_copy)
            """,
            "drive uniqueness",
        ),
        (
            "source-tenant-transitive-whole-row-helper",
            """
            CREATE FUNCTION omninode_internal.source_tenant_key(
                omninode_internal.runtime_state
            ) RETURNS uuid LANGUAGE sql IMMUTABLE STRICT AS $$
                SELECT $1.source_tenant_id
            $$;
            CREATE FUNCTION omninode_internal.nested_source_tenant_key(
                omninode_internal.runtime_state
            ) RETURNS uuid LANGUAGE sql IMMUTABLE STRICT AS $$
                SELECT omninode_internal.source_tenant_key
                    /* comments are PostgreSQL whitespace */ ($1)
            $$;
            CREATE UNIQUE INDEX runtime_state_transitive_source_guard
            ON omninode_internal.runtime_state (
                omninode_internal.nested_source_tenant_key(runtime_state.*)
            );
            CREATE FUNCTION omninode_internal.nested_source_tenant_guard()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                PERFORM omninode_internal.nested_source_tenant_key(NEW);
                RETURN NEW;
            END
            $$;
            CREATE TRIGGER runtime_state_transitive_source_guard
            BEFORE INSERT OR UPDATE ON omninode_internal.runtime_state
            FOR EACH ROW EXECUTE FUNCTION
                omninode_internal.nested_source_tenant_guard();
            CREATE VIEW omninode_internal.runtime_state_transitive_authority AS
            SELECT state_id,
                   omninode_internal.nested_source_tenant_key(runtime_state.*)
                       AS source_key
            FROM omninode_internal.runtime_state
            """,
            ("drive uniqueness", "drive write eligibility", "drive authorization"),
        ),
        (
            "source-tenant-named-whole-row-helper",
            """
            CREATE FUNCTION omninode_internal.named_runtime_state_digest(
                row_value omninode_internal.runtime_state
            ) RETURNS text LANGUAGE sql IMMUTABLE STRICT AS $$
                SELECT pg_catalog.md5(row_value::text)
            $$;
            CREATE UNIQUE INDEX runtime_state_named_whole_row_guard
            ON omninode_internal.runtime_state (
                omninode_internal.named_runtime_state_digest(runtime_state.*)
            );
            CREATE FUNCTION omninode_internal.named_whole_row_guard()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                PERFORM omninode_internal.named_runtime_state_digest(NEW);
                RETURN NEW;
            END
            $$;
            CREATE TRIGGER runtime_state_named_whole_row_guard
            BEFORE INSERT OR UPDATE ON omninode_internal.runtime_state
            FOR EACH ROW EXECUTE FUNCTION
                omninode_internal.named_whole_row_guard();
            CREATE VIEW omninode_internal.runtime_state_named_whole_row_authority AS
            SELECT state_id,
                   omninode_internal.named_runtime_state_digest(runtime_state.*)
                       AS row_digest
            FROM omninode_internal.runtime_state
            """,
            ("drive uniqueness", "drive write eligibility", "drive authorization"),
        ),
        (
            "source-tenant-trigger-body",
            """
            CREATE FUNCTION omninode_internal.reject_source_tenant()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.source_tenant_id IS NOT NULL THEN
                    RAISE EXCEPTION 'source tenant cannot select writes';
                END IF;
                RETURN NEW;
            END
            $$;
            CREATE TRIGGER runtime_state_source_guard
            BEFORE INSERT OR UPDATE ON omninode_internal.runtime_state
            FOR EACH ROW EXECUTE FUNCTION omninode_internal.reject_source_tenant()
            """,
            "drive write eligibility",
        ),
        (
            "source-tenant-dependent-view",
            """
            CREATE VIEW omninode_internal.runtime_state_authorized AS
            SELECT state_id, source_tenant_id, payload
            FROM omninode_internal.runtime_state
            WHERE source_tenant_id = current_setting('app.tenant_id', true)::uuid
            """,
            "drive authorization",
        ),
    )
    for control, statement, expected in controls:
        with admin.cursor() as cursor:
            cursor.execute(statement)
        state = _table_state(admin, declaration)
        violations = validate_application_database_relation_states((state,), _TOPOLOGY)
        if isinstance(expected, tuple):
            missing = tuple(
                fragment
                for fragment in expected
                if not any(fragment in violation for violation in violations)
            )
            if missing:
                raise AssertionError(
                    f"seeded control {control!r} did not discriminate {missing!r}: "
                    + "; ".join(violations)
                )
            print(f"domain_control={control} status=PASS")
        else:
            _assert_red(control, violations, expected)
        admin.rollback()
    return len(controls)


def _run_live_function_definition_control(
    admin: psycopg2.extensions.connection,
    declaration: ModelApplicationRelationDeclaration,
    evidence: ModelApplicationDatabaseTenantIsolationEvidence | None,
) -> int:
    """Prove non-body routine metadata drift changes the audited fingerprint."""
    admin.commit()
    with admin.cursor() as cursor:
        cursor.execute("ALTER FUNCTION tenant.safe_report() STABLE")
    state = _function_state(admin, declaration, evidence)
    _assert_red(
        "security-definer-volatility-drift",
        validate_application_database_relation_states((state,), _TOPOLOGY),
        "does not match the audited definition hash",
    )
    admin.rollback()
    return 1


def _run_live_policy_role_control(
    admin: psycopg2.extensions.connection,
    tenant: ModelApplicationDatabaseRelationState,
) -> int:
    """Prove exact pg_policy.polroles drift is observed and rejected."""
    canonical_policy = tenant.canonical_policy_name
    if canonical_policy is None:
        raise AssertionError("green tenant state is missing its canonical policy name")
    declaration = tenant.declaration
    admin.commit()
    with admin.cursor() as cursor:
        cursor.execute(
            pg_sql.SQL("ALTER POLICY {} ON {}.{} TO {}").format(
                pg_sql.Identifier(canonical_policy),
                pg_sql.Identifier(declaration.schema),
                pg_sql.Identifier(declaration.name),
                pg_sql.Identifier("omninode_runtime"),
            )
        )
    state = _table_state(admin, declaration)
    _assert_red(
        "canonical-policy-unrelated-role",
        validate_application_database_relation_states((state,), _TOPOLOGY),
        "role scope",
    )
    admin.rollback()
    return 1


def _run_live_identity_root_role_controls(
    admin: psycopg2.extensions.connection,
    identity_root: ModelApplicationDatabaseRelationState,
) -> int:
    """Seed real role reachability and prove both catalog and SET ROLE controls."""
    root_control = identity_root.identity_root_control_state
    if root_control is None:
        raise AssertionError("green identity-root control state is missing")
    runtime_principal = _DATABASE.bindings["onex_api"].principal
    bridge_role = "seeded_identity_root_bridge"

    with admin.cursor() as cursor:
        cursor.execute("SAVEPOINT identity_root_membership_control")
        try:
            cursor.execute(
                pg_sql.SQL("CREATE ROLE {} NOLOGIN").format(
                    pg_sql.Identifier(bridge_role)
                )
            )
            cursor.execute(
                pg_sql.SQL("GRANT {} TO {}").format(
                    pg_sql.Identifier(root_control.role),
                    pg_sql.Identifier(bridge_role),
                )
            )
            cursor.execute(
                pg_sql.SQL("GRANT {} TO {}").format(
                    pg_sql.Identifier(bridge_role),
                    pg_sql.Identifier(runtime_principal),
                )
            )
            membership_principals = _runtime_identity_root_membership_principals(
                admin,
                root_control.role,
            )
        finally:
            cursor.execute("ROLLBACK TO SAVEPOINT identity_root_membership_control")
            cursor.execute("RELEASE SAVEPOINT identity_root_membership_control")
    _assert_red(
        "identity-root-runtime-membership",
        validate_application_database_relation_states(
            (
                identity_root.model_copy(
                    update={
                        "identity_root_control_state": root_control.model_copy(
                            update={
                                "runtime_membership_principals": (membership_principals)
                            }
                        )
                    }
                ),
            ),
            _TOPOLOGY,
        ),
        "membership path",
    )
    admin.commit()

    try:
        with admin.cursor() as cursor:
            cursor.execute(
                pg_sql.SQL("GRANT {} TO {}").format(
                    pg_sql.Identifier(root_control.role),
                    pg_sql.Identifier(runtime_principal),
                )
            )
        admin.commit()
        denied_principals = _runtime_identity_root_set_role_denials(root_control.role)
    finally:
        admin.rollback()
        with admin.cursor() as cursor:
            cursor.execute(
                pg_sql.SQL("REVOKE {} FROM {}").format(
                    pg_sql.Identifier(root_control.role),
                    pg_sql.Identifier(runtime_principal),
                )
            )
        admin.commit()
    restored_memberships = _runtime_identity_root_membership_principals(
        admin,
        root_control.role,
    )
    restored_denials = _runtime_identity_root_set_role_denials(root_control.role)
    expected_runtime_principals = tuple(
        sorted(binding.principal for binding in _DATABASE.bindings.values())
    )
    if restored_memberships or restored_denials != expected_runtime_principals:
        raise AssertionError(
            "identity-root role-control RED proof did not restore the green "
            "runtime authority state"
        )
    _assert_red(
        "identity-root-runtime-set-role",
        validate_application_database_relation_states(
            (
                identity_root.model_copy(
                    update={
                        "identity_root_control_state": root_control.model_copy(
                            update={
                                "runtime_set_role_denied_principals": (
                                    denied_principals
                                )
                            }
                        )
                    }
                ),
            ),
            _TOPOLOGY,
        ),
        "SET ROLE denial",
    )
    return 2


def _run_cross_gate_controls(
    identities: tuple[ModelApplicationDatabasePoolIdentity, ...],
) -> int:
    sql_controls: Mapping[str, tuple[str, str]] = {
        "public-table": ("CREATE TABLE public.events (id uuid);", "public"),
        "unqualified-table": (
            "CREATE TABLE events (id uuid);",
            "schema-qualified",
        ),
        "unknown-schema": (
            "CREATE TABLE mystery.events (id uuid);",
            "unknown topology schema",
        ),
        "unqualified-read": ("SELECT * FROM events;", "schema-qualified"),
        "unqualified-merge": (
            "MERGE INTO events USING tenant.incoming ON false WHEN NOT MATCHED THEN DO NOTHING;",
            "schema-qualified",
        ),
        "unqualified-grant": (
            "GRANT SELECT ON TABLE events TO app_dashboard;",
            "schema-qualified",
        ),
        "unqualified-foreign-key": (
            "CREATE TABLE tenant.child (id uuid REFERENCES parent(id));",
            "schema-qualified",
        ),
    }
    for control, (statement, expected) in sql_controls.items():
        _assert_red(
            control,
            lint_application_database_sql(statement, _TOPOLOGY),
            expected,
        )

    wrong_database = identities[0].model_copy(
        update={"current_database": "omninode_cloud"}
    )
    _assert_red(
        "old-application-database",
        validate_application_database_pool_identities(
            (wrong_database, *identities[1:]), _TOPOLOGY
        ),
        "one physical database",
    )
    duplicate_user = identities[1].model_copy(
        update={"current_user": identities[0].current_user}
    )
    _assert_red(
        "duplicate-pool-user",
        validate_application_database_pool_identities(
            (identities[0], duplicate_user, *identities[2:]), _TOPOLOGY
        ),
        "distinct",
    )
    return len(sql_controls) + 2


def main() -> None:
    admin = _connect(_ADMIN_DSN)
    try:
        observed_catalog = _catalog_identities(admin)
        catalog_authority = load_application_database_ownership_identities(
            (_OWNERSHIP_MANIFEST,)
        )
        declarations = _authoritative_declarations(observed_catalog)
        relation_states = tuple(
            _relation_state(admin, declaration) for declaration in declarations
        )
        states_by_name = {
            (state.declaration.schema, state.declaration.name): state
            for state in relation_states
        }
        tenant = states_by_name[("tenant", "events")]
        identity_root = states_by_name[("tenant", "tenants")]
        internal = states_by_name[("omninode_internal", "runtime_state")]
        view = states_by_name[("tenant", "events_view")]
        function = states_by_name[("tenant", "safe_report")]

        relation_violations = validate_application_database_catalog_census(
            relation_states,
            observed_catalog,
            _TOPOLOGY,
            authoritative_identities=catalog_authority,
        )
        if relation_violations:
            raise AssertionError(
                f"green exact relation catalog failed: {relation_violations}"
            )
        control_count = _run_catalog_controls(
            admin,
            relation_states,
            observed_catalog,
            catalog_authority,
        )
        control_count += _run_live_source_tenant_controls(
            admin,
            internal.declaration,
        )
        function_state = function.function_state
        if function_state is None:
            raise AssertionError("green function state is missing")
        control_count += _run_live_function_definition_control(
            admin,
            function.declaration,
            function_state.tenant_isolation_evidence,
        )
        control_count += _run_live_policy_role_control(admin, tenant)
        control_count += _run_live_identity_root_role_controls(
            admin,
            identity_root,
        )
    finally:
        admin.close()

    identities = _pool_identities()
    pool_violations = validate_application_database_pool_identities(
        identities, _TOPOLOGY
    )
    if pool_violations:
        raise AssertionError(f"green pool identity proof failed: {pool_violations}")
    if lint_application_database_sql(
        "CREATE TABLE tenant.seeded_green (id uuid PRIMARY KEY);", _TOPOLOGY
    ):
        raise AssertionError("qualified green migration SQL was rejected")

    control_count += _run_relation_controls(
        identity_root,
        tenant,
        internal,
        view,
        function,
    )
    control_count += _run_cross_gate_controls(identities)
    print(
        "application_domain_enforcement_status=PASS "
        f"postgres_major=16 relations={len(relation_states)} "
        f"catalog_objects={len(observed_catalog)} pools={len(identities)} "
        f"red_controls={control_count}"
    )


if __name__ == "__main__":
    main()
