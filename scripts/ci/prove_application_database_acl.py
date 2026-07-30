# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Rebuilt PostgreSQL 16 ACL/default-privilege and rollback proof."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

import psycopg2
import psycopg2.extras
import yaml
from psycopg2 import sql

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.validation.application_database_acl import (
    PUBLIC_PRINCIPAL,
    build_application_database_acl_matrix,
    render_application_database_acl_sql,
)
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.enums.enum_application_database_acl_render_phase import (
    EnumApplicationDatabaseAclRenderPhase,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
    ModelApplicationDatabaseAclSource,
)
from omnibase_infra.validation.models.model_application_database_acl_policy import (
    ModelApplicationDatabaseAclPolicy,
)
from omnibase_infra.validation.models.model_application_database_acl_row import (
    ModelApplicationDatabaseAclRow,
)
from omnibase_infra.validation.models.model_application_database_default_acl_row import (
    ModelApplicationDatabaseDefaultAclRow,
)
from omnibase_infra.validation.models.model_application_database_observed_role_state import (
    ModelApplicationDatabaseObservedRoleState,
)
from omnibase_infra.validation.models.model_application_database_principal_inventory import (
    ModelApplicationDatabasePrincipalInventory,
)
from omnibase_infra.validation.models.model_application_database_role_membership import (
    ModelApplicationDatabaseRoleMembership,
)
from omnibase_infra.validation.models.model_application_database_role_state import (
    ModelApplicationDatabaseRoleState,
)
from omnibase_infra.validation.models.model_application_relation_evidence_inventory import (
    ModelApplicationRelationEvidenceInventory,
)

ADMIN_DSN = os.environ["ADMIN_DSN"]
DATABASE = os.environ.get("PROOF_DATABASE", "omnidash_analytics")
DATABASE_HOST = os.environ.get("PROOF_DB_HOST", "postgres")
DATABASE_PORT = int(os.environ.get("PROOF_DB_PORT", "5432"))
ROLE_PASSWORD = "acl-proof-only"  # pragma: allowlist secret
MANAGED_SCHEMAS = ("tenant", "omninode_internal", "platform_catalog")
MUTATED_SCHEMAS = (*MANAGED_SCHEMAS, "public")
LEGACY_DEFAULT_SCHEMA = "legacy_acl_sentinel"
OWNERS = (
    "owner_onex_tenant",
    "owner_omninode_internal",
    "owner_platform_catalog",
)
WORKLOADS = (
    "app_dashboard",
    "omninode_runtime",
    "onex_api",
    "tenant_projection_writer",
)
SERVICE_DATABASE_ROLES = {
    "keycloak": "keycloak_service",
    "omnibase_infra": "omnibase_infra_service",
    "omninode_cloud": "omninode_cloud_service",
    "omniclaude": "omniclaude_service",
    "omniintelligence": "omniintelligence_service",
    "omnimemory": "omnimemory_service",
    "umami": "umami_service",
}
EXPECTED_PRECHANGE = Path(
    os.environ.get(
        "EXPECTED_PRECHANGE",
        "/app/proof/prechange-fixture-acl.json",
    )
)
OBSERVED_PRECHANGE = Path(
    os.environ.get(
        "OBSERVED_PRECHANGE",
        "/output/observed-prechange-acl.json",
    )
)
FIXTURES = Path(os.environ.get("ACL_FIXTURES", "/app/proof/fixtures"))


class _AclSnapshot(TypedDict):
    """Typed shape of the durable synthetic pre-change ACL artifact."""

    schema_version: str
    provenance: dict[str, object]
    roles: list[dict[str, object]]
    memberships: list[dict[str, object]]
    database_owner: list[dict[str, object]]
    database_acl: list[dict[str, object]]
    schema_owners: list[dict[str, object]]
    schema_acl: list[dict[str, object]]
    object_owners: list[dict[str, object]]
    object_acl: list[dict[str, object]]
    column_acl: list[dict[str, object]]
    default_acl: list[dict[str, object]]
    default_acl_catalog_rows: list[dict[str, object]]


def _admin(dsn: str = ADMIN_DSN) -> psycopg2.extensions.connection:
    connection = psycopg2.connect(dsn)
    connection.autocommit = True
    return connection


def _admin_dsn_for_database(database: str) -> str:
    """Retarget the configured admin DSN without dropping authentication."""
    parameters = psycopg2.extensions.parse_dsn(ADMIN_DSN)
    parameters["dbname"] = database
    return cast("str", psycopg2.extensions.make_dsn(**parameters))


def _dict_rows(
    statement: str,
    parameters: Sequence[object] = (),
    *,
    dsn: str = ADMIN_DSN,
) -> list[dict[str, object]]:
    connection = _admin(dsn)
    try:
        with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(statement, parameters)
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def _sorted_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    materialized = [dict(row) for row in rows]
    return sorted(
        materialized,
        key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
    )


def _capture_legacy_default_acl(
    *,
    dsn: str = ADMIN_DSN,
) -> list[dict[str, object]]:
    """Capture the unrelated-schema default ACL sentinel exactly."""
    return _sorted_rows(
        _dict_rows(
            """
            SELECT owner.rolname AS owner, namespace.nspname AS schema_name,
                   defaults.defaclobjtype AS object_type,
                   array_to_string(
                     ARRAY(
                       SELECT acl_item::text
                       FROM unnest(defaults.defaclacl) acl_item
                       ORDER BY acl_item::text
                     ),
                     ','
                   ) AS raw_acl
            FROM pg_default_acl defaults
            JOIN pg_roles owner ON owner.oid = defaults.defaclrole
            JOIN pg_namespace namespace ON namespace.oid = defaults.defaclnamespace
            WHERE namespace.nspname = %s
            """,
            (LEGACY_DEFAULT_SCHEMA,),
            dsn=dsn,
        )
    )


def _postgres_major() -> int:
    row = _dict_rows("SHOW server_version_num")[0]
    return int(str(row["server_version_num"])) // 10_000


def _matrix_principals(matrix: ModelApplicationDatabaseAclMatrix) -> set[str]:
    return {
        principal
        for principals in (
            *matrix.declared_principals.values(),
            *matrix.observed_principals.values(),
            *matrix.absent_principals.values(),
            *matrix.observed_connect_principals.values(),
            *matrix.absent_connect_principals.values(),
            *matrix.allowed_connect_principals.values(),
        )
        for principal in principals
    }


def _owner_roles(matrix: ModelApplicationDatabaseAclMatrix) -> set[str]:
    return (
        set(matrix.database_owners.values())
        | {obj.owner for obj in matrix.objects}
        | {row.owner for row in matrix.default_privileges}
    )


def _allowed_connect_roles(matrix: ModelApplicationDatabaseAclMatrix) -> set[str]:
    return {
        principal
        for principals in matrix.allowed_connect_principals.values()
        for principal in principals
    }


def _controlled_roles(matrix: ModelApplicationDatabaseAclMatrix) -> set[str]:
    return (
        _owner_roles(matrix)
        | _matrix_principals(matrix)
        | {membership.role for membership in matrix.allowed_memberships}
        | {membership.member for membership in matrix.allowed_memberships}
    )


def _membership_member_roles(matrix: ModelApplicationDatabaseAclMatrix) -> set[str]:
    """Roles whose parent edges are explicitly reconciled by the renderer."""
    return (
        _matrix_principals(matrix)
        .union(_owner_roles(matrix))
        .difference(matrix.retained_administrative_principals)
    )


def _protected_parent_roles(matrix: ModelApplicationDatabaseAclMatrix) -> set[str]:
    """Roles whose inheritance by any undeclared child breaks the allowlist."""
    return {
        state.role for state in matrix.governed_role_states if state.manage_memberships
    }


def _capture_snapshot(
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    dsn: str = ADMIN_DSN,
) -> _AclSnapshot:
    """Capture every fixture ACL input needed by rollback mechanics."""

    def query(
        statement: str,
        parameters: Sequence[object] = (),
    ) -> list[dict[str, object]]:
        return _dict_rows(statement, parameters, dsn=dsn)

    owner_names = sorted(_owner_roles(matrix))
    controlled_roles = sorted(_controlled_roles(matrix))
    protected_parent_roles = sorted(_protected_parent_roles(matrix))
    membership_member_roles = sorted(_membership_member_roles(matrix))
    managed_schema_names = list(MANAGED_SCHEMAS)
    mutated_schema_names = list(MUTATED_SCHEMAS)
    roles = query(
        """
        SELECT rolname, rolcanlogin, rolsuper, rolinherit, rolcreaterole,
               rolcreatedb, rolreplication, rolbypassrls
        FROM pg_roles
        WHERE rolname = ANY(%s)
        """,
        (controlled_roles,),
    )
    memberships = query(
        """
        SELECT parent.rolname AS parent_role, member.rolname AS member_role,
               grantor.rolname AS grantor,
               membership.admin_option, membership.inherit_option,
               membership.set_option
        FROM pg_auth_members membership
        JOIN pg_roles parent ON parent.oid = membership.roleid
        JOIN pg_roles member ON member.oid = membership.member
        JOIN pg_roles grantor ON grantor.oid = membership.grantor
        WHERE member.rolname = ANY(%s)
           OR parent.rolname = ANY(%s)
        """,
        (membership_member_roles, protected_parent_roles),
    )
    database_owner = query(
        """
        SELECT database.datname AS database_name, owner.rolname AS owner
        FROM pg_database database
        JOIN pg_roles owner ON owner.oid = database.datdba
        WHERE database.datname = ANY(%s)
        """,
        (list(matrix.required_connect_databases),),
    )
    database_acl = query(
        """
        SELECT database.datname AS database_name,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               grantor.rolname AS grantor, acl.privilege_type,
               acl.is_grantable
        FROM pg_database database
        CROSS JOIN LATERAL aclexplode(
          COALESCE(database.datacl, acldefault('d', database.datdba))
        ) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        JOIN pg_roles grantor ON grantor.oid = acl.grantor
        WHERE database.datname = ANY(%s)
          AND acl.grantee <> database.datdba
        """,
        (list(matrix.required_connect_databases),),
    )
    schema_owners = query(
        """
        SELECT namespace.nspname AS schema_name, owner.rolname AS owner
        FROM pg_namespace namespace
        JOIN pg_roles owner ON owner.oid = namespace.nspowner
        WHERE namespace.nspname = ANY(%s)
        """,
        (mutated_schema_names,),
    )
    schema_acl = query(
        """
        SELECT namespace.nspname AS schema_name,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               grantor.rolname AS grantor, acl.privilege_type,
               acl.is_grantable
        FROM pg_namespace namespace
        CROSS JOIN LATERAL aclexplode(
          COALESCE(namespace.nspacl, acldefault('n', namespace.nspowner))
        ) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        JOIN pg_roles grantor ON grantor.oid = acl.grantor
        WHERE namespace.nspname = ANY(%s)
          AND acl.grantee <> namespace.nspowner
        """,
        (mutated_schema_names,),
    )
    relation_owners = query(
        """
        SELECT namespace.nspname AS schema_name, relation.relname AS object_name,
               CASE relation.relkind
                 WHEN 'r' THEN 'table'
                 WHEN 'p' THEN 'table'
                 WHEN 'v' THEN 'view'
                 WHEN 'm' THEN 'materialized_view'
                 WHEN 'S' THEN 'sequence'
               END AS catalog_kind,
               CASE relation.relkind
                 WHEN 'v' THEN 'VIEW'
                 WHEN 'm' THEN 'MATERIALIZED VIEW'
                 WHEN 'S' THEN 'SEQUENCE'
                 ELSE 'TABLE'
               END AS owner_keyword,
               CASE WHEN relation.relkind = 'S' THEN 'SEQUENCE' ELSE 'TABLE' END
                 AS object_type,
               owner.rolname AS owner
        FROM pg_class relation
        JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
        JOIN pg_roles owner ON owner.oid = relation.relowner
        WHERE namespace.nspname = ANY(%s)
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'S')
        """,
        (managed_schema_names,),
    )
    relation_acl = query(
        """
        SELECT namespace.nspname AS schema_name, relation.relname AS object_name,
               CASE WHEN relation.relkind = 'S' THEN 'SEQUENCE' ELSE 'TABLE' END
                 AS object_type,
               CASE relation.relkind
                 WHEN 'r' THEN 'table'
                 WHEN 'p' THEN 'table'
                 WHEN 'v' THEN 'view'
                 WHEN 'm' THEN 'materialized_view'
                 WHEN 'S' THEN 'sequence'
               END AS catalog_kind,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               grantor.rolname AS grantor, acl.privilege_type,
               acl.is_grantable
        FROM pg_class relation
        JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
        CROSS JOIN LATERAL aclexplode(
          COALESCE(
            relation.relacl,
            acldefault(
              (CASE WHEN relation.relkind = 'S' THEN 's' ELSE 'r' END)::"char",
              relation.relowner
            )
          )
        ) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        JOIN pg_roles grantor ON grantor.oid = acl.grantor
        WHERE namespace.nspname = ANY(%s)
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'S')
          AND acl.grantee <> relation.relowner
        """,
        (managed_schema_names,),
    )
    routine_owners = query(
        """
        SELECT namespace.nspname AS schema_name, procedure.proname AS object_name,
               '(' || pg_get_function_identity_arguments(procedure.oid) || ')'
                 AS function_signature,
               CASE WHEN procedure.prokind = 'p' THEN 'procedure' ELSE 'function' END
                 AS catalog_kind,
               CASE WHEN procedure.prokind = 'p' THEN 'PROCEDURE' ELSE 'FUNCTION' END
                 AS owner_keyword,
               CASE WHEN procedure.prokind = 'p' THEN 'PROCEDURE' ELSE 'FUNCTION' END
                 AS object_type,
               owner.rolname AS owner
        FROM pg_proc procedure
        JOIN pg_namespace namespace ON namespace.oid = procedure.pronamespace
        JOIN pg_roles owner ON owner.oid = procedure.proowner
        WHERE namespace.nspname = ANY(%s) AND procedure.prokind IN ('f', 'p')
        """,
        (managed_schema_names,),
    )
    routine_acl = query(
        """
        SELECT namespace.nspname AS schema_name, procedure.proname AS object_name,
               '(' || pg_get_function_identity_arguments(procedure.oid) || ')'
                 AS function_signature,
               CASE WHEN procedure.prokind = 'p' THEN 'PROCEDURE' ELSE 'FUNCTION' END
                 AS object_type,
               CASE WHEN procedure.prokind = 'p' THEN 'procedure' ELSE 'function' END
                 AS catalog_kind,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               grantor.rolname AS grantor, acl.privilege_type,
               acl.is_grantable
        FROM pg_proc procedure
        JOIN pg_namespace namespace ON namespace.oid = procedure.pronamespace
        CROSS JOIN LATERAL aclexplode(
          COALESCE(procedure.proacl, acldefault('f', procedure.proowner))
        ) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        JOIN pg_roles grantor ON grantor.oid = acl.grantor
        WHERE namespace.nspname = ANY(%s) AND procedure.prokind IN ('f', 'p')
          AND acl.grantee <> procedure.proowner
        """,
        (managed_schema_names,),
    )
    type_owners = query(
        """
        SELECT namespace.nspname AS schema_name, type.typname AS object_name,
               'type' AS catalog_kind, 'TYPE' AS owner_keyword,
               'TYPE' AS object_type,
               owner.rolname AS owner
        FROM pg_type type
        JOIN pg_namespace namespace ON namespace.oid = type.typnamespace
        JOIN pg_roles owner ON owner.oid = type.typowner
        LEFT JOIN pg_class relation ON relation.oid = type.typrelid
        WHERE namespace.nspname = ANY(%s)
          AND type.typisdefined
          AND (
            (type.typtype IN ('b', 'd', 'e', 'r', 'm') AND type.typelem = 0)
            OR (type.typtype = 'c' AND relation.relkind = 'c')
          )
        """,
        (managed_schema_names,),
    )
    type_acl = query(
        """
        SELECT namespace.nspname AS schema_name, type.typname AS object_name,
               'TYPE' AS object_type, 'type' AS catalog_kind,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               grantor.rolname AS grantor, acl.privilege_type,
               acl.is_grantable
        FROM pg_type type
        JOIN pg_namespace namespace ON namespace.oid = type.typnamespace
        LEFT JOIN pg_class relation ON relation.oid = type.typrelid
        CROSS JOIN LATERAL aclexplode(
          COALESCE(type.typacl, acldefault('T', type.typowner))
        ) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        JOIN pg_roles grantor ON grantor.oid = acl.grantor
        WHERE namespace.nspname = ANY(%s)
          AND type.typisdefined
          AND (
            (type.typtype IN ('b', 'd', 'e', 'r', 'm') AND type.typelem = 0)
            OR (type.typtype = 'c' AND relation.relkind = 'c')
          )
          AND acl.grantee <> type.typowner
        """,
        (managed_schema_names,),
    )
    column_acl = query(
        """
        SELECT namespace.nspname AS schema_name,
               relation.relname AS object_name,
               CASE relation.relkind
                 WHEN 'r' THEN 'table'
                 WHEN 'p' THEN 'table'
                 WHEN 'v' THEN 'view'
                 WHEN 'm' THEN 'materialized_view'
               END AS catalog_kind,
               attribute.attname AS column_name,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               grantor.rolname AS grantor, acl.privilege_type,
               acl.is_grantable
        FROM pg_attribute attribute
        JOIN pg_class relation ON relation.oid = attribute.attrelid
        JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
        CROSS JOIN LATERAL aclexplode(attribute.attacl) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        JOIN pg_roles grantor ON grantor.oid = acl.grantor
        WHERE namespace.nspname = ANY(%s)
          AND relation.relkind IN ('r', 'p', 'v', 'm')
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped
          AND attribute.attacl IS NOT NULL
          AND acl.grantee <> relation.relowner
        """,
        (managed_schema_names,),
    )
    default_acl = query(
        """
        SELECT owner.rolname AS owner,
               namespace.nspname AS schema_name,
               CASE defaults.defaclobjtype
                 WHEN 'r' THEN 'TABLE'
                 WHEN 'S' THEN 'SEQUENCE'
                 WHEN 'f' THEN 'FUNCTION'
                 WHEN 'T' THEN 'TYPE'
               END AS object_type,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               grantor.rolname AS grantor, acl.privilege_type,
               acl.is_grantable
        FROM pg_default_acl defaults
        JOIN pg_roles owner ON owner.oid = defaults.defaclrole
        LEFT JOIN pg_namespace namespace ON namespace.oid = defaults.defaclnamespace
        CROSS JOIN LATERAL aclexplode(defaults.defaclacl) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        JOIN pg_roles grantor ON grantor.oid = acl.grantor
        WHERE owner.rolname = ANY(%s)
          AND (defaults.defaclnamespace = 0 OR namespace.nspname = ANY(%s))
          AND acl.grantee <> defaults.defaclrole
        """,
        (owner_names, managed_schema_names),
    )
    default_acl_catalog_rows = query(
        """
        SELECT owner.rolname AS owner,
               namespace.nspname AS schema_name,
               CASE defaults.defaclobjtype
                 WHEN 'r' THEN 'TABLE'
                 WHEN 'S' THEN 'SEQUENCE'
                 WHEN 'f' THEN 'FUNCTION'
                 WHEN 'T' THEN 'TYPE'
               END AS object_type,
               array_to_string(
                 ARRAY(
                   SELECT acl_item::text
                   FROM unnest(defaults.defaclacl) acl_item
                   ORDER BY acl_item::text
                 ),
                 ','
               ) AS raw_acl
        FROM pg_default_acl defaults
        JOIN pg_roles owner ON owner.oid = defaults.defaclrole
        LEFT JOIN pg_namespace namespace ON namespace.oid = defaults.defaclnamespace
        WHERE owner.rolname = ANY(%s)
          AND (defaults.defaclnamespace = 0 OR namespace.nspname = ANY(%s))
        """,
        (owner_names, managed_schema_names),
    )
    return {
        "schema_version": "3.0",
        "provenance": {
            "source": "sanitized_postgresql_16_fixture",
            "live_database_read": False,
            "dump_derived": False,
            "authorization_scope": "synthetic_proof",
            "postgres_major": _postgres_major(),
        },
        "roles": _sorted_rows(roles),
        "memberships": _sorted_rows(memberships),
        "database_owner": _sorted_rows(database_owner),
        "database_acl": _sorted_rows(database_acl),
        "schema_owners": _sorted_rows(schema_owners),
        "schema_acl": _sorted_rows(schema_acl),
        "object_owners": _sorted_rows(
            [
                *relation_owners,
                *routine_owners,
                *type_owners,
            ]
        ),
        "object_acl": _sorted_rows([*relation_acl, *routine_acl, *type_acl]),
        "column_acl": _sorted_rows(column_acl),
        "default_acl": _sorted_rows(default_acl),
        "default_acl_catalog_rows": _sorted_rows(default_acl_catalog_rows),
    }


def _fixture_matrix() -> ModelApplicationDatabaseAclMatrix:
    topology = ModelDeploymentTopology.from_yaml(FIXTURES / "topology.yaml")
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(
        yaml.safe_load((FIXTURES / "inventory.yaml").read_text(encoding="utf-8"))
    )
    principal_inventory = ModelApplicationDatabasePrincipalInventory.model_validate(
        yaml.safe_load(
            (FIXTURES / "principal-inventory-postgres16.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    external_principal_inventories = tuple(
        ModelApplicationDatabasePrincipalInventory.model_validate(item)
        for item in yaml.safe_load(
            (FIXTURES / "principal-inventories-external.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    acl_policy = ModelApplicationDatabaseAclPolicy.model_validate(
        yaml.safe_load(
            (FIXTURES / "acl-policy-postgres16.yaml").read_text(encoding="utf-8")
        )
    )
    external_inventory_sources = tuple(
        ModelApplicationDatabaseAclSource(
            source_key=f"synthetic_principal_inventory_{inventory.database_ref}",
            repository="synthetic/docker-proof",
            revision=f"{index:x}" * 40,
            path="proof/fixtures/principal-inventories-external.yaml",
            sha256=f"{index:x}" * 64,
            purpose="principal_inventory",
        )
        for index, inventory in enumerate(external_principal_inventories, start=3)
    )
    sources = (
        ModelApplicationDatabaseAclSource(
            source_key="synthetic_topology",
            repository="synthetic/docker-proof",
            revision="a" * 40,
            path="proof/fixtures/topology.yaml",
            sha256="b" * 64,
            purpose="topology",
        ),
        ModelApplicationDatabaseAclSource(
            source_key="synthetic_inventory",
            repository="synthetic/docker-proof",
            revision="c" * 40,
            path="proof/fixtures/inventory.yaml",
            sha256="d" * 64,
            purpose="relation_inventory",
        ),
        ModelApplicationDatabaseAclSource(
            source_key="synthetic_principal_inventory",
            repository="synthetic/docker-proof",
            revision="e" * 40,
            path="proof/fixtures/principal-inventory-postgres16.yaml",
            sha256="f" * 64,
            purpose="principal_inventory",
        ),
        ModelApplicationDatabaseAclSource(
            source_key="synthetic_acl_policy",
            repository="synthetic/docker-proof",
            revision="1" * 40,
            path="proof/fixtures/acl-policy-postgres16.yaml",
            sha256="2" * 64,
            purpose="acl_policy",
        ),
        *external_inventory_sources,
    )
    principal_inventories = {
        "synthetic_principal_inventory": principal_inventory,
        **{
            f"synthetic_principal_inventory_{inventory.database_ref}": inventory
            for inventory in external_principal_inventories
        },
    }
    return build_application_database_acl_matrix(
        topology=topology,
        sources=sources,
        relation_inventories={"synthetic_inventory": inventory},
        service_manifests={},
        principal_inventories=principal_inventories,
        acl_policies={"synthetic_acl_policy": acl_policy},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
        required_connect_databases=tuple(
            policy.physical_database for policy in acl_policy.connection_policies
        ),
    )


def _expected_acl_set(
    matrix: ModelApplicationDatabaseAclMatrix,
) -> set[tuple[object, ...]]:
    objects = {obj.identity: obj for obj in matrix.objects}
    expected: set[tuple[object, ...]] = {
        (
            (
                "PROCEDURE"
                if row.object_type is EnumDatabaseGrantObjectType.FUNCTION
                and objects[
                    (
                        row.database_ref,
                        row.schema_ref or "",
                        row.object_type,
                        row.object_ref or "",
                        row.function_signature,
                    )
                ].catalog_kind
                == "procedure"
                else row.object_type.value
            ),
            row.physical_database,
            row.schema_ref or "",
            row.object_ref or "",
            row.function_signature or "",
            row.principal,
            privilege.value,
            False,
        )
        for row in matrix.rows
        for privilege in row.privileges
    }
    expected.update(
        (
            "DATABASE",
            physical_database,
            "",
            "",
            "",
            principal,
            "CONNECT",
            False,
        )
        for physical_database, principals in matrix.allowed_connect_principals.items()
        for principal in principals
    )
    return expected


def _actual_acl_set(
    snapshot: _AclSnapshot,
    application_database: str,
) -> set[tuple[object, ...]]:
    result: set[tuple[object, ...]] = set()
    for row in snapshot["database_acl"]:
        result.add(
            (
                "DATABASE",
                str(row["database_name"]),
                "",
                "",
                "",
                str(row["grantee"]),
                str(row["privilege_type"]),
                bool(row["is_grantable"]),
            )
        )
    for row in snapshot["schema_acl"]:
        result.add(
            (
                "SCHEMA",
                application_database,
                str(row["schema_name"]),
                "",
                "",
                str(row["grantee"]),
                str(row["privilege_type"]),
                bool(row["is_grantable"]),
            )
        )
    for row in snapshot["object_acl"]:
        result.add(
            (
                str(row["object_type"]),
                application_database,
                str(row["schema_name"]),
                str(row["object_name"]),
                str(row.get("function_signature") or ""),
                str(row["grantee"]),
                str(row["privilege_type"]),
                bool(row["is_grantable"]),
            )
        )
    return result


def _expected_hardened_default_acl_catalog_rows(
    matrix: ModelApplicationDatabaseAclMatrix,
) -> list[dict[str, object]]:
    """Project the exact explicit rows needed to remove builtin PUBLIC defaults."""
    rows: list[dict[str, object]] = []
    for owner in sorted(_owner_roles(matrix)):
        rows.extend(
            (
                {
                    "owner": owner,
                    "schema_name": None,
                    "object_type": "FUNCTION",
                    "raw_acl": f"{owner}=X/{owner}",
                },
                {
                    "owner": owner,
                    "schema_name": None,
                    "object_type": "TYPE",
                    "raw_acl": f"{owner}=U/{owner}",
                },
            )
        )
    return _sorted_rows(rows)


def _catalog_violations(
    snapshot: _AclSnapshot,
    matrix: ModelApplicationDatabaseAclMatrix,
) -> set[str]:
    violations: set[str] = set()
    expected = _expected_acl_set(matrix)
    application_databases = {row.physical_database for row in matrix.default_privileges}
    assert len(application_databases) == 1, application_databases
    actual = _actual_acl_set(snapshot, next(iter(application_databases)))
    unexpected = actual - expected
    missing = expected - actual
    if missing:
        violations.add("MISSING_DECLARED_PRIVILEGE")
    if any(row[5] == PUBLIC_PRINCIPAL for row in unexpected):
        violations.add("PUBLIC_PRIVILEGE")
    if any(bool(row[7]) for row in actual):
        violations.add("GRANT_OPTION")
    if any(
        row[0] == "DATABASE" and row[6] in {"CREATE", "TEMPORARY"} for row in unexpected
    ):
        violations.add("DATABASE_DDL_PRIVILEGE")
    if snapshot["column_acl"]:
        violations.add("COLUMN_PRIVILEGE")

    declared_principals = {
        principal
        for principals in matrix.declared_principals.values()
        for principal in principals
    }
    if any(
        row[5] not in declared_principals | {PUBLIC_PRINCIPAL} for row in unexpected
    ):
        violations.add("UNDECLARED_PRINCIPAL_PRIVILEGE")

    governed_roles = {state.role for state in matrix.governed_role_states}
    owner_roles = _owner_roles(matrix)
    governed_login_roles = {
        state.role for state in matrix.governed_role_states if state.login
    }
    object_domains = {
        (
            obj.object_type.value,
            obj.schema_ref,
            obj.object_ref,
            obj.function_signature or "",
        ): obj.domain
        for obj in matrix.objects
    }
    database_ref_by_physical = {
        row.physical_database: row.database_ref
        for row in matrix.rows
        if row.object_type is EnumDatabaseGrantObjectType.DATABASE
    }
    for row in unexpected:
        object_type, _, schema_name, object_name, signature, grantee, _, _ = row
        if object_type == "SCHEMA":
            domain = matrix.schema_domains.get(
                database_ref_by_physical.get(str(row[1]), ""), {}
            ).get(str(schema_name))
        else:
            normalized_object_type = str(
                "FUNCTION" if object_type == "PROCEDURE" else object_type
            )
            domain = object_domains.get(
                (
                    normalized_object_type,
                    str(schema_name),
                    str(object_name),
                    str(signature),
                )
            )
        if domain is not None and domain not in matrix.principal_domains.get(
            str(grantee), ()
        ):
            violations.add("CROSS_DOMAIN_PRIVILEGE")
    unexpected_objects = Counter(
        (row[5], row[2], row[3], row[4])
        for row in unexpected
        if row[0] in {"TABLE", "SEQUENCE", "FUNCTION", "PROCEDURE", "TYPE"}
        and row[5] != PUBLIC_PRINCIPAL
    )
    if len(unexpected_objects) >= 2 or any(
        count > 1 for count in unexpected_objects.values()
    ):
        violations.add("BROAD_GRANT")
    elif unexpected:
        violations.add("UNEXPECTED_PRIVILEGE")

    if snapshot["default_acl"] or snapshot[
        "default_acl_catalog_rows"
    ] != _expected_hardened_default_acl_catalog_rows(matrix):
        violations.add("UNSAFE_DEFAULT_PRIVILEGE")
    expected_owners = {
        (
            obj.catalog_kind,
            obj.schema_ref,
            obj.object_ref,
            obj.function_signature or "",
        ): obj.owner
        for obj in matrix.objects
    }
    actual_owners = {
        (
            str(row["catalog_kind"]),
            str(row["schema_name"]),
            str(row["object_name"]),
            str(row.get("function_signature") or ""),
        ): str(row["owner"])
        for row in snapshot["object_owners"]
    }
    if actual_owners != expected_owners:
        violations.add("OWNER_MISMATCH")
    expected_schema_owners = {
        (row.schema_ref, row.owner) for row in matrix.default_privileges
    }
    actual_schema_owners = {
        (str(row["schema_name"]), str(row["owner"]))
        for row in snapshot["schema_owners"]
        if str(row["schema_name"]) in MANAGED_SCHEMAS
    }
    if expected_schema_owners != actual_schema_owners:
        violations.add("OWNER_MISMATCH")

    expected_database_owners = dict(matrix.observed_connect_database_owners)
    expected_database_owners.update(
        {
            row.physical_database: matrix.database_owners[row.database_ref]
            for row in matrix.rows
            if row.object_type is EnumDatabaseGrantObjectType.DATABASE
        }
    )
    actual_database_owners = {
        str(row["database_name"]): str(row["owner"])
        for row in snapshot["database_owner"]
    }
    if actual_database_owners != expected_database_owners:
        violations.add("OWNER_MISMATCH")

    runtime_owned = {
        str(row["owner"])
        for row in (
            *snapshot["schema_owners"],
            *snapshot["object_owners"],
        )
        if str(row["owner"]) in declared_principals
    }
    application_database_owned = {
        str(row["owner"])
        for row in snapshot["database_owner"]
        if str(row["database_name"]) == DATABASE
        and str(row["owner"]) in declared_principals
    }
    runtime_owned.update(application_database_owned)
    if runtime_owned:
        violations.add("RUNTIME_OWNERSHIP")

    for role in snapshot["roles"]:
        role_name = str(role["rolname"])
        if role_name not in governed_roles:
            continue
        if (
            (role_name in governed_login_roles and not bool(role["rolcanlogin"]))
            or (role_name in owner_roles and bool(role["rolcanlogin"]))
            or bool(role["rolsuper"])
            or bool(role["rolinherit"])
            or bool(role["rolcreaterole"])
            or bool(role["rolcreatedb"])
            or bool(role["rolreplication"])
            or bool(role["rolbypassrls"])
        ):
            violations.add("RUNTIME_ROLE_ATTRIBUTES")

    expected_memberships = {
        (
            membership.role,
            membership.member,
            membership.admin_option,
            membership.inherit_option,
            membership.set_option,
        )
        for membership in matrix.allowed_memberships
    }
    actual_memberships = {
        (
            str(row["parent_role"]),
            str(row["member_role"]),
            bool(row["admin_option"]),
            bool(row["inherit_option"]),
            bool(row["set_option"]),
        )
        for row in snapshot["memberships"]
    }
    if actual_memberships - expected_memberships:
        violations.add("OWNER_MEMBERSHIP")
    if expected_memberships - actual_memberships:
        violations.add("MISSING_DECLARED_MEMBERSHIP")
    return violations


def _execute_matrix(
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    dsn: str = ADMIN_DSN,
    phase: EnumApplicationDatabaseAclRenderPhase = (
        EnumApplicationDatabaseAclRenderPhase.FULL
    ),
) -> None:
    rendered = render_application_database_acl_sql(
        matrix,
        allow_synthetic_proof=True,
        phase=phase,
    )
    result = subprocess.run(
        ["psql", "--no-psqlrc", "--dbname", dsn],
        input=rendered,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "psql ACL apply failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _atomic_failure_proof(
    matrix: ModelApplicationDatabaseAclMatrix,
    snapshot: _AclSnapshot,
    *,
    dsn: str = ADMIN_DSN,
    phase: EnumApplicationDatabaseAclRenderPhase = (
        EnumApplicationDatabaseAclRenderPhase.FULL
    ),
) -> None:
    """Prove the emitted psql script rolls back all prior statements on error."""
    rendered = render_application_database_acl_sql(
        matrix,
        allow_synthetic_proof=True,
        phase=phase,
    )
    injected = rendered.replace(
        "COMMIT;\n",
        "SELECT 1 / 0; -- injected atomicity control\nCOMMIT;\n",
    )
    assert injected != rendered
    result = subprocess.run(
        ["psql", "--no-psqlrc", "--dbname", dsn],
        input=injected,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert _capture_snapshot(matrix, dsn=dsn) == snapshot, (
        "failed psql application leaked a partial ACL mutation"
    )
    print("acl_phase=psql_atomic_failure red_control=ROLLBACK status=PASS")


def _identifier_list(values: Sequence[str]) -> sql.Composed:
    return sql.SQL(", ").join(sql.Identifier(value) for value in values)


def _grant_rows(
    cursor: psycopg2.extensions.cursor,
    rows: Sequence[Mapping[str, object]],
    *,
    target: Callable[[Mapping[str, object]], sql.Composable],
    prefix: Callable[[Mapping[str, object]], sql.Composable],
) -> None:
    grouped: dict[tuple[str, str, str, bool], set[str]] = defaultdict(set)
    row_by_key: dict[tuple[str, str, str, bool], Mapping[str, object]] = {}
    for row in rows:
        target_identity = {
            key_name: value
            for key_name, value in row.items()
            if key_name not in {"privilege_type", "grantee", "grantor", "is_grantable"}
        }
        group_key = (
            json.dumps(target_identity, sort_keys=True),
            str(row["grantee"]),
            str(row["grantor"]),
            bool(row["is_grantable"]),
        )
        grouped[group_key].add(str(row["privilege_type"]))
        row_by_key[group_key] = row

    remaining = set(grouped)
    ordered: list[tuple[str, str, str, bool]] = []
    while remaining:
        ready = sorted(
            key
            for key in remaining
            if not any(
                key[0] == prerequisite[0]
                and key[2] == prerequisite[1]
                and prerequisite[3]
                and bool(grouped[key] & grouped[prerequisite])
                for prerequisite in remaining
                if prerequisite != key
            )
        )
        if not ready:
            raise AssertionError("ACL grantor graph is cyclic or lacks a grant option")
        ordered.extend(ready)
        remaining.difference_update(ready)

    for key in ordered:
        privileges = grouped[key]
        row = row_by_key[key]
        cursor.execute(
            sql.SQL("SET ROLE {}").format(sql.Identifier(str(row["grantor"])))
        )
        cursor.execute(
            sql.SQL("GRANT {} ON {} TO {}{}").format(
                sql.SQL(", ").join(sql.SQL(item) for item in sorted(privileges)),
                prefix(row) + target(row),
                sql.SQL("PUBLIC")
                if row["grantee"] == PUBLIC_PRINCIPAL
                else sql.Identifier(str(row["grantee"])),
                sql.SQL(" WITH GRANT OPTION") if row["is_grantable"] else sql.SQL(""),
            )
        )
        cursor.execute("RESET ROLE")


def _restore_memberships(
    cursor: psycopg2.extensions.cursor,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Restore exact PG16 membership grantors after their ADMIN paths exist."""
    remaining = {json.dumps(dict(row), sort_keys=True) for row in rows}
    row_by_key = {json.dumps(dict(row), sort_keys=True): row for row in rows}
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            key
            for key in remaining
            if not any(
                str(row_by_key[key]["parent_role"])
                == str(row_by_key[prerequisite]["parent_role"])
                and str(row_by_key[key]["grantor"])
                == str(row_by_key[prerequisite]["member_role"])
                and bool(row_by_key[prerequisite]["admin_option"])
                for prerequisite in remaining
                if prerequisite != key
            )
        )
        if not ready:
            raise AssertionError(
                "role membership grantor graph is cyclic or lacks ADMIN OPTION"
            )
        ordered.extend(ready)
        remaining.difference_update(ready)

    for key in ordered:
        row = row_by_key[key]
        cursor.execute(
            sql.SQL("SET ROLE {}").format(sql.Identifier(str(row["grantor"])))
        )
        cursor.execute(
            sql.SQL("GRANT {} TO {} WITH ADMIN {}, INHERIT {}, SET {}").format(
                sql.Identifier(str(row["parent_role"])),
                sql.Identifier(str(row["member_role"])),
                sql.SQL("TRUE" if row["admin_option"] else "FALSE"),
                sql.SQL("TRUE" if row["inherit_option"] else "FALSE"),
                sql.SQL("TRUE" if row["set_option"] else "FALSE"),
            )
        )
        cursor.execute("RESET ROLE")


def _object_target(row: Mapping[str, object]) -> sql.Composable:
    qualified = sql.SQL("{}.{}").format(
        sql.Identifier(str(row["schema_name"])),
        sql.Identifier(str(row["object_name"])),
    )
    if row["object_type"] in {"FUNCTION", "PROCEDURE"}:
        return qualified + sql.SQL(str(row.get("function_signature", "()")))
    return qualified


def _object_prefix(row: Mapping[str, object]) -> sql.Composable:
    return sql.SQL(str(row["object_type"]) + " ")


def _grant_column_rows(
    cursor: psycopg2.extensions.cursor,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Restore exact per-column ACLs, including grantor and grant option."""
    grouped: dict[tuple[str, str, str, bool], set[str]] = defaultdict(set)
    row_by_key: dict[tuple[str, str, str, bool], Mapping[str, object]] = {}
    for row in rows:
        target_identity = json.dumps(
            {
                "schema_name": row["schema_name"],
                "object_name": row["object_name"],
                "column_name": row["column_name"],
            },
            sort_keys=True,
        )
        key = (
            target_identity,
            str(row["grantee"]),
            str(row["grantor"]),
            bool(row["is_grantable"]),
        )
        grouped[key].add(str(row["privilege_type"]))
        row_by_key[key] = row
    remaining = set(grouped)
    ordered: list[tuple[str, str, str, bool]] = []
    while remaining:
        ready = sorted(
            key
            for key in remaining
            if not any(
                key[0] == prerequisite[0]
                and key[2] == prerequisite[1]
                and prerequisite[3]
                and bool(grouped[key] & grouped[prerequisite])
                for prerequisite in remaining
                if prerequisite != key
            )
        )
        if not ready:
            raise AssertionError(
                "column ACL grantor graph is cyclic or lacks a grant option"
            )
        ordered.extend(ready)
        remaining.difference_update(ready)

    for key in ordered:
        row = row_by_key[key]
        grantee = (
            sql.SQL("PUBLIC")
            if row["grantee"] == PUBLIC_PRINCIPAL
            else sql.Identifier(str(row["grantee"]))
        )
        cursor.execute(
            sql.SQL("SET ROLE {}").format(sql.Identifier(str(row["grantor"])))
        )
        cursor.execute(
            sql.SQL("GRANT {} ON TABLE {}.{} TO {}{}").format(
                sql.SQL(", ").join(
                    sql.SQL("{} ({})").format(
                        sql.SQL(privilege),
                        sql.Identifier(str(row["column_name"])),
                    )
                    for privilege in sorted(grouped[key])
                ),
                sql.Identifier(str(row["schema_name"])),
                sql.Identifier(str(row["object_name"])),
                grantee,
                sql.SQL(" WITH GRANT OPTION") if row["is_grantable"] else sql.SQL(""),
            )
        )
        cursor.execute("RESET ROLE")


def _reset_target_default_acls(
    cursor: psycopg2.extensions.cursor,
    *,
    owners: Sequence[str],
    schemas: Sequence[str],
) -> None:
    """Return governed global/schema defaults to PostgreSQL built-ins."""
    cursor.execute(
        """
        SELECT owner.rolname AS owner, namespace.nspname AS schema_name,
               CASE defaults.defaclobjtype
                 WHEN 'r' THEN 'TABLES'
                 WHEN 'S' THEN 'SEQUENCES'
                 WHEN 'f' THEN 'FUNCTIONS'
                 WHEN 'T' THEN 'TYPES'
               END AS object_keyword,
               COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
               acl.privilege_type
        FROM pg_default_acl defaults
        JOIN pg_roles owner ON owner.oid = defaults.defaclrole
        LEFT JOIN pg_namespace namespace ON namespace.oid = defaults.defaclnamespace
        CROSS JOIN LATERAL aclexplode(defaults.defaclacl) acl
        LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
        WHERE owner.rolname = ANY(%s)
          AND (defaults.defaclnamespace = 0 OR namespace.nspname = ANY(%s))
          AND acl.grantee <> defaults.defaclrole
        ORDER BY owner.rolname, namespace.nspname NULLS FIRST,
                 defaults.defaclobjtype, grantee.rolname NULLS FIRST,
                 acl.privilege_type
        """,
        (list(owners), list(schemas)),
    )
    for owner, schema_name, object_keyword, grantee, privilege in cursor.fetchall():
        grantee_sql = (
            sql.SQL("PUBLIC")
            if str(grantee) == PUBLIC_PRINCIPAL
            else sql.Identifier(str(grantee))
        )
        scope = (
            sql.SQL(" IN SCHEMA {}").format(sql.Identifier(str(schema_name)))
            if schema_name is not None
            else sql.SQL("")
        )
        cursor.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {}{} REVOKE {} ON {} FROM {} CASCADE"
            ).format(
                sql.Identifier(str(owner)),
                scope,
                sql.SQL(str(privilege)),
                sql.SQL(str(object_keyword)),
                grantee_sql,
            )
        )

    # Global function/type defaults include PUBLIC by definition. Table and
    # sequence defaults do not; grant-then-revoke removes any explicit row that
    # merely encodes the built-in state.
    for owner in owners:
        owner_identifier = sql.Identifier(owner)
        cursor.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} "
                "GRANT EXECUTE ON FUNCTIONS TO PUBLIC"
            ).format(owner_identifier)
        )
        cursor.execute(
            sql.SQL(
                "ALTER DEFAULT PRIVILEGES FOR ROLE {} GRANT USAGE ON TYPES TO PUBLIC"
            ).format(owner_identifier)
        )
        for object_keyword in ("TABLES", "SEQUENCES"):
            cursor.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} "
                    f"GRANT ALL PRIVILEGES ON {object_keyword} TO PUBLIC"
                ).format(owner_identifier)
            )
            cursor.execute(
                sql.SQL(
                    "ALTER DEFAULT PRIVILEGES FOR ROLE {} "
                    f"REVOKE ALL PRIVILEGES ON {object_keyword} FROM PUBLIC"
                ).format(owner_identifier)
            )


def _restore_snapshot(
    snapshot: _AclSnapshot,
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    inject_failure: bool = False,
    dsn: str = ADMIN_DSN,
) -> None:
    """Atomically restore only from the durable pre-change catalog snapshot."""
    matrix_principals = _matrix_principals(matrix)
    snapshot_grantees = {
        str(row["grantee"])
        for rows in (
            snapshot["database_acl"],
            snapshot["schema_acl"],
            snapshot["object_acl"],
            snapshot["column_acl"],
            snapshot["default_acl"],
        )
        for row in rows
        if str(row["grantee"]) != PUBLIC_PRINCIPAL
    }
    membership_roles = {
        str(row[key])
        for row in snapshot["memberships"]
        for key in ("parent_role", "member_role", "grantor")
    }
    revocation_principals = sorted(
        matrix_principals | snapshot_grantees | membership_roles
    )
    database_revocation_principals = revocation_principals
    membership_members = sorted(_membership_member_roles(matrix))
    protected_parents = sorted(_protected_parent_roles(matrix))
    snapshot_role_names = {str(row["rolname"]) for row in snapshot["roles"]}
    renderer_created_roles = sorted(
        (
            _owner_roles(matrix)
            | {
                principal
                for principals in matrix.declared_principals.values()
                for principal in principals
            }
            | _allowed_connect_roles(matrix)
        )
        - snapshot_role_names
    )
    snapshot_schema_names = {
        str(row["schema_name"]) for row in snapshot["schema_owners"]
    }
    renderer_created_schemas = sorted(set(MANAGED_SCHEMAS) - snapshot_schema_names)
    connection = psycopg2.connect(dsn)
    try:
        with connection.cursor() as cursor:
            grantees = _identifier_list(revocation_principals)
            database_grantees = _identifier_list(database_revocation_principals)
            for database_name in matrix.required_connect_databases:
                cursor.execute(
                    sql.SQL(
                        "REVOKE ALL PRIVILEGES ON DATABASE {} FROM PUBLIC, {} CASCADE"
                    ).format(
                        sql.Identifier(database_name),
                        database_grantees,
                    )
                )
            for schema_name in MUTATED_SCHEMAS:
                cursor.execute(
                    sql.SQL(
                        "REVOKE ALL PRIVILEGES ON SCHEMA {} FROM PUBLIC, {} CASCADE"
                    ).format(sql.Identifier(schema_name), grantees)
                )
            for obj in matrix.objects:
                keyword = {
                    "table": "TABLE",
                    "view": "TABLE",
                    "materialized_view": "TABLE",
                    "sequence": "SEQUENCE",
                    "function": "FUNCTION",
                    "procedure": "PROCEDURE",
                    "type": "TYPE",
                }[obj.catalog_kind]
                target = sql.SQL("{}.{}").format(
                    sql.Identifier(obj.schema_ref), sql.Identifier(obj.object_ref)
                )
                if obj.object_type is EnumDatabaseGrantObjectType.FUNCTION:
                    target += sql.SQL(obj.function_signature or "()")
                cursor.execute(
                    sql.SQL(
                        f"REVOKE ALL PRIVILEGES ON {keyword} {{}} FROM PUBLIC, {{}} CASCADE"
                    ).format(target, grantees)
                )
            default_owners = sorted(_owner_roles(matrix))
            _reset_target_default_acls(
                cursor,
                owners=default_owners,
                schemas=MANAGED_SCHEMAS,
            )

            if inject_failure:
                cursor.execute("SELECT 1 / 0")

            for row in snapshot["roles"]:
                attributes = [
                    "LOGIN" if row["rolcanlogin"] else "NOLOGIN",
                    "SUPERUSER" if row["rolsuper"] else "NOSUPERUSER",
                    "INHERIT" if row["rolinherit"] else "NOINHERIT",
                    "CREATEROLE" if row["rolcreaterole"] else "NOCREATEROLE",
                    "CREATEDB" if row["rolcreatedb"] else "NOCREATEDB",
                    "REPLICATION" if row["rolreplication"] else "NOREPLICATION",
                    "BYPASSRLS" if row["rolbypassrls"] else "NOBYPASSRLS",
                ]
                cursor.execute(
                    sql.SQL("ALTER ROLE {} " + " ".join(attributes)).format(
                        sql.Identifier(str(row["rolname"]))
                    )
                )
            cursor.execute(
                """
                SELECT parent.rolname, child.rolname, grantor.rolname
                FROM pg_auth_members membership
                JOIN pg_roles parent ON parent.oid = membership.roleid
                JOIN pg_roles child ON child.oid = membership.member
                JOIN pg_roles grantor ON grantor.oid = membership.grantor
                WHERE child.rolname = ANY(%s)
                   OR parent.rolname = ANY(%s)
                """,
                (membership_members, protected_parents),
            )
            current_memberships = cursor.fetchall()
            for parent, member, grantor in current_memberships:
                cursor.execute(
                    sql.SQL("REVOKE {} FROM {} GRANTED BY {} CASCADE").format(
                        sql.Identifier(str(parent)),
                        sql.Identifier(str(member)),
                        sql.Identifier(str(grantor)),
                    )
                )
            _restore_memberships(cursor, snapshot["memberships"])
            for row in snapshot["database_owner"]:
                cursor.execute(
                    sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                        sql.Identifier(str(row["database_name"])),
                        sql.Identifier(str(row["owner"])),
                    )
                )
            for row in snapshot["schema_owners"]:
                cursor.execute(
                    sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(
                        sql.Identifier(str(row["schema_name"])),
                        sql.Identifier(str(row["owner"])),
                    )
                )
            owner_restore_order = {
                "TABLE": 0,
                "VIEW": 1,
                "MATERIALIZED VIEW": 2,
                "SEQUENCE": 3,
                "TYPE": 4,
                "FUNCTION": 5,
                "PROCEDURE": 6,
            }
            for row in sorted(
                snapshot["object_owners"],
                key=lambda item: (
                    owner_restore_order[str(item["owner_keyword"])],
                    str(item["schema_name"]),
                    str(item["object_name"]),
                ),
            ):
                keyword = str(row["owner_keyword"])
                target = _object_target(row)
                cursor.execute(
                    sql.SQL(f"ALTER {keyword} {{}} OWNER TO {{}}").format(
                        target, sql.Identifier(str(row["owner"]))
                    )
                )

            _grant_rows(
                cursor,
                snapshot["database_acl"],
                target=lambda row: sql.Identifier(str(row["database_name"])),
                prefix=lambda row: sql.SQL("DATABASE "),
            )
            _grant_rows(
                cursor,
                snapshot["schema_acl"],
                target=lambda row: sql.Identifier(str(row["schema_name"])),
                prefix=lambda row: sql.SQL("SCHEMA "),
            )
            _grant_rows(
                cursor,
                snapshot["object_acl"],
                target=_object_target,
                prefix=_object_prefix,
            )
            _grant_column_rows(cursor, snapshot["column_acl"])
            raw_default_rows = {
                (
                    str(row["owner"]),
                    row["schema_name"],
                    str(row["object_type"]),
                ): str(row["raw_acl"])
                for row in snapshot["default_acl_catalog_rows"]
            }
            for owner in default_owners:
                for object_type, privilege, marker in (
                    ("FUNCTION", "EXECUTE", "=X"),
                    ("TYPE", "USAGE", "=U"),
                ):
                    raw_acl = raw_default_rows.get((owner, None, object_type))
                    public_builtin_present = raw_acl is not None and any(
                        item.startswith(marker) for item in raw_acl.split(",")
                    )
                    if raw_acl is not None and not public_builtin_present:
                        cursor.execute(
                            sql.SQL(
                                "ALTER DEFAULT PRIVILEGES FOR ROLE {} "
                                f"REVOKE {privilege} ON {object_type}S FROM PUBLIC"
                            ).format(sql.Identifier(owner))
                        )
            default_rows = snapshot["default_acl"]
            for row in default_rows:
                if row["grantor"] != row["owner"]:
                    raise AssertionError(
                        "default ACL grantor must equal its owning role for exact restore"
                    )
                grantee = (
                    sql.SQL("PUBLIC")
                    if row["grantee"] == PUBLIC_PRINCIPAL
                    else sql.Identifier(str(row["grantee"]))
                )
                keyword = {
                    "TABLE": "TABLES",
                    "SEQUENCE": "SEQUENCES",
                    "FUNCTION": "FUNCTIONS",
                    "TYPE": "TYPES",
                }[str(row["object_type"])]
                default_schema_name = row["schema_name"]
                scope = (
                    sql.SQL(" IN SCHEMA {} ").format(
                        sql.Identifier(str(default_schema_name))
                    )
                    if default_schema_name is not None
                    else sql.SQL(" ")
                )
                cursor.execute(
                    sql.SQL("SET ROLE {}").format(sql.Identifier(str(row["owner"])))
                )
                cursor.execute(
                    sql.SQL(
                        f"ALTER DEFAULT PRIVILEGES{{}}"
                        f"GRANT {row['privilege_type']} ON {keyword} TO {{}}{{}}"
                    ).format(
                        scope,
                        grantee,
                        sql.SQL(" WITH GRANT OPTION")
                        if row["is_grantable"]
                        else sql.SQL(""),
                    )
                )
                cursor.execute("RESET ROLE")
            for schema_name in renderer_created_schemas:
                cursor.execute(
                    sql.SQL("DROP SCHEMA {}").format(sql.Identifier(schema_name))
                )
            for role_name in renderer_created_roles:
                # Return newly-created owners to PostgreSQL's built-in default ACL.
                # DROP ROLE then remains a fail-closed emptiness/dependency check:
                # it cannot cascade into a relation that appeared after apply.
                cursor.execute(
                    sql.SQL(
                        "ALTER DEFAULT PRIVILEGES FOR ROLE {} "
                        "GRANT EXECUTE ON FUNCTIONS TO PUBLIC"
                    ).format(sql.Identifier(role_name))
                )
                cursor.execute(
                    sql.SQL(
                        "ALTER DEFAULT PRIVILEGES FOR ROLE {} "
                        "GRANT USAGE ON TYPES TO PUBLIC"
                    ).format(sql.Identifier(role_name))
                )
                cursor.execute(
                    sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name))
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _role_connection(
    role: str,
    database: str = DATABASE,
) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=DATABASE_HOST,
        port=DATABASE_PORT,
        dbname=database,
        user=role,
        password=ROLE_PASSWORD,
    )


def _expect_denied(connection: psycopg2.extensions.connection, statement: str) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute(statement)
    except psycopg2.Error:
        connection.rollback()
        return
    raise AssertionError(f"Expected PostgreSQL denial: {statement}")


def _expect_connection_denied(role: str, database: str) -> None:
    try:
        connection = _role_connection(role, database)
    except psycopg2.OperationalError:
        return
    connection.close()
    raise AssertionError(f"{role} unexpectedly connected to {database}")


def _legacy_default_behavior_proof() -> None:
    """Prove the unrelated-schema default grant still affects future objects."""
    connection = psycopg2.connect(ADMIN_DSN)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE owner_onex_tenant")
            cursor.execute(
                "CREATE TABLE legacy_acl_sentinel.default_acl_probe(id integer)"
            )
            cursor.execute("RESET ROLE")
            cursor.execute(
                "SELECT has_table_privilege(%s, %s, 'SELECT')",
                (
                    "untrusted_login",
                    "legacy_acl_sentinel.default_acl_probe",
                ),
            )
            assert cursor.fetchone() == (True,)
    finally:
        connection.rollback()
        connection.close()


def _builtin_default_rollback_proof() -> None:
    """Prove rollback restores PostgreSQL's implicit PUBLIC routine/type ACLs."""
    connection = psycopg2.connect(ADMIN_DSN)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE owner_onex_tenant")
            cursor.execute(
                "CREATE FUNCTION legacy_acl_sentinel.rollback_function() "
                "RETURNS integer LANGUAGE sql AS $$SELECT 1$$"
            )
            cursor.execute(
                "CREATE TYPE legacy_acl_sentinel.rollback_type AS ENUM ('restored')"
            )
            cursor.execute("RESET ROLE")
            cursor.execute(
                """
                SELECT has_function_privilege(
                         'untrusted_login',
                         'legacy_acl_sentinel.rollback_function()',
                         'EXECUTE'
                       ),
                       has_type_privilege(
                         'untrusted_login',
                         'legacy_acl_sentinel.rollback_type',
                         'USAGE'
                       )
                """
            )
            assert cursor.fetchone() == (True, True)
    finally:
        connection.rollback()
        connection.close()


def _legacy_scaffold_behavior_proof() -> None:
    """Prove P1 preserves a pre-existing login's CONNECT/read/write path."""
    connection = _role_connection("legacy_probe_login", "acl_scaffold_probe")
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM public.legacy_scaffold_data")
            assert cursor.fetchall() == [(1,)]
            cursor.execute("INSERT INTO public.legacy_scaffold_data VALUES (2)")
    finally:
        connection.rollback()
        connection.close()


def _connection_isolation_proof(matrix: ModelApplicationDatabaseAclMatrix) -> None:
    """Exercise positive and cross-database negative CONNECT cases."""
    assert set(matrix.required_connect_databases) == {
        DATABASE,
        *SERVICE_DATABASE_ROLES,
    }
    all_service_roles = set(SERVICE_DATABASE_ROLES.values())
    for database, allowed_role in SERVICE_DATABASE_ROLES.items():
        allowed = _role_connection(allowed_role, database)
        allowed.close()
        for denied_role in sorted(
            set(WORKLOADS) | all_service_roles | {"shadow_login", "untrusted_login"}
        ):
            if denied_role != allowed_role:
                _expect_connection_denied(denied_role, database)
    for service_role in sorted(all_service_roles):
        _expect_connection_denied(service_role, DATABASE)
    print("acl_phase=eight_database_connect_isolation status=PASS")


def _behavioral_proof(matrix: ModelApplicationDatabaseAclMatrix) -> None:
    tenant = _role_connection("tenant_projection_writer")
    dashboard = _role_connection("app_dashboard")
    api = _role_connection("onex_api")
    runtime = _role_connection("omninode_runtime")
    try:
        with tenant.cursor() as cursor:
            cursor.execute(
                "INSERT INTO tenant.delegation_events (payload) VALUES ('matrix')"
            )
            cursor.execute("CALL tenant.record_delegation('procedure')")
        tenant.commit()
        _expect_denied(tenant, "SELECT * FROM omninode_internal.runtime_state")
        _expect_denied(tenant, "CREATE TABLE tenant.illegal(id integer)")

        with dashboard.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM tenant.delegation_events")
            assert cursor.fetchone()[0] >= 1
            cursor.execute("SELECT display_name FROM tenant.tenant_account_names")
            assert cursor.fetchone()[0] == "Synthetic"
            cursor.execute("SELECT code FROM platform_catalog.plan_tier_snapshot")
            assert cursor.fetchone()[0] == "beta"
        _expect_denied(
            dashboard,
            "INSERT INTO tenant.delegation_events (payload) VALUES ('forbidden')",
        )
        _expect_denied(dashboard, "SELECT * FROM omninode_internal.runtime_state")

        with api.cursor() as cursor:
            cursor.execute(
                "INSERT INTO tenant.tenant_accounts VALUES (2, 'API matrix')"
            )
            cursor.execute("SELECT code FROM platform_catalog.plan_tiers")
            assert cursor.fetchone()[0] == "beta"
            cursor.execute("SELECT ROW(1, 'API')::tenant.account_ref")
            cursor.execute("SELECT '[1,2)'::tenant.account_id_span")
            cursor.execute("SELECT '{}'::tenant.account_id_span_set")
        api.commit()
        _expect_denied(api, "UPDATE platform_catalog.plan_tiers SET code='x'")
        _expect_denied(api, "SELECT * FROM omninode_internal.runtime_state")

        with runtime.cursor() as cursor:
            cursor.execute(
                "INSERT INTO omninode_internal.runtime_state VALUES (2, 'blocked')"
            )
            cursor.execute("SELECT 'runtime'::omninode_internal.runtime_code")
        runtime.commit()
        _expect_denied(runtime, "SELECT * FROM tenant.delegation_events")
        _expect_denied(runtime, "SELECT * FROM platform_catalog.plan_tiers")
    finally:
        tenant.close()
        dashboard.close()
        api.close()
        runtime.close()

    try:
        untrusted = _role_connection("untrusted_login")
    except psycopg2.OperationalError:
        pass
    else:
        untrusted.close()
        raise AssertionError("PUBLIC CONNECT was not revoked")
    _connection_isolation_proof(matrix)


def _migration_membership_proof() -> None:
    inherited = _dict_rows(
        """
        SELECT has_table_privilege(
          'db_migrator', 'tenant.tenant_accounts', 'SELECT'
        ) AS inherited
        """
    )[0]
    assert not bool(inherited["inherited"]), inherited
    connection = _admin()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION AUTHORIZATION db_migrator")
            cursor.execute("SET ROLE owner_onex_tenant")
            cursor.execute("SELECT count(*) FROM tenant.tenant_accounts")
            assert cursor.fetchone()[0] >= 1
            cursor.execute("RESET SESSION AUTHORIZATION")
    finally:
        connection.close()


def _future_default_proof() -> None:
    connection = _admin()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE owner_onex_tenant")
            cursor.execute("CREATE TABLE tenant.future_table(id integer)")
            cursor.execute("CREATE SEQUENCE tenant.future_sequence")
            cursor.execute(
                "CREATE FUNCTION tenant.future_function() RETURNS integer "
                "LANGUAGE sql AS $$SELECT 1$$"
            )
            cursor.execute("CREATE TYPE tenant.future_type AS ENUM ('future')")
            cursor.execute("RESET ROLE")
        for role in (*WORKLOADS, "untrusted_login"):
            checks = _dict_rows(
                """
                SELECT has_table_privilege(%s, 'tenant.future_table', 'SELECT') AS table_ok,
                       has_sequence_privilege(%s, 'tenant.future_sequence', 'USAGE') AS sequence_ok,
                       has_function_privilege(%s, 'tenant.future_function()', 'EXECUTE') AS function_ok,
                       has_type_privilege(%s, 'tenant.future_type', 'USAGE') AS type_ok
                """,
                (role, role, role, role),
            )[0]
            assert not any(bool(value) for value in checks.values()), (role, checks)
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE tenant.future_table")
            cursor.execute("DROP SEQUENCE tenant.future_sequence")
            cursor.execute("DROP FUNCTION tenant.future_function()")
            cursor.execute("DROP TYPE tenant.future_type")
    finally:
        connection.close()


def _extra_object_rejection_proof(
    matrix: ModelApplicationDatabaseAclMatrix,
) -> None:
    """Prove an undeclared runtime-owned object is a hard catalog violation."""
    connection = _admin()
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE TABLE tenant.undeclared_runtime_owned(id integer)")
            cursor.execute(
                "ALTER TABLE tenant.undeclared_runtime_owned OWNER TO app_dashboard"
            )
        violations = _catalog_violations(_capture_snapshot(matrix), matrix)
        assert {"OWNER_MISMATCH", "RUNTIME_OWNERSHIP"} <= violations, violations
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE tenant.undeclared_runtime_owned")
    finally:
        connection.close()
    print("acl_phase=extra_object_red status=DETECTED")


def _unknown_acl_and_membership_reconciliation_proof(
    matrix: ModelApplicationDatabaseAclMatrix,
    green: _AclSnapshot,
) -> None:
    """Prove live catalog sweeps remove grantees/children absent from evidence."""
    connection = _admin()
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE ROLE hostile_unknown")
            cursor.execute(
                "GRANT SELECT ON tenant.partitioned_events "
                "TO rls_admin WITH GRANT OPTION"
            )
            cursor.execute("GRANT USAGE ON SCHEMA tenant TO rls_admin")
            cursor.execute("SET ROLE rls_admin")
            cursor.execute(
                "GRANT SELECT ON tenant.partitioned_events TO hostile_unknown"
            )
            cursor.execute("RESET ROLE")
            cursor.execute("REVOKE USAGE ON SCHEMA tenant FROM rls_admin")
            cursor.execute("GRANT owner_onex_tenant TO hostile_unknown")
            cursor.execute(
                "ALTER DEFAULT PRIVILEGES FOR ROLE owner_onex_tenant "
                "IN SCHEMA tenant GRANT SELECT ON TABLES TO hostile_unknown"
            )
        _execute_matrix(matrix)
        assert _capture_snapshot(matrix) == green
        with connection.cursor() as cursor:
            cursor.execute("DROP ROLE hostile_unknown")
    finally:
        connection.close()
    print("acl_phase=unknown_catalog_rows_reconciled status=PASS")


def _scaffold_probe_matrix(
    source_matrix: ModelApplicationDatabaseAclMatrix,
) -> ModelApplicationDatabaseAclMatrix:
    """Build a typed scaffold whose desired roles and schemas are all absent."""
    database_ref = "probe"
    physical_database = "acl_scaffold_probe"
    principals_by_domain = {
        EnumDatabaseSchemaDomain.TENANT: "probe_tenant_writer",
        EnumDatabaseSchemaDomain.OMNINODE_INTERNAL: "probe_internal_writer",
        EnumDatabaseSchemaDomain.PLATFORM_CATALOG: "probe_catalog_reader",
    }
    schema_domains = {
        "tenant": EnumDatabaseSchemaDomain.TENANT,
        "omninode_internal": EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,
        "platform_catalog": EnumDatabaseSchemaDomain.PLATFORM_CATALOG,
    }
    owners = {
        "tenant": "probe_owner_tenant",
        "omninode_internal": "probe_owner_internal",
        "platform_catalog": "probe_owner_catalog",
    }
    declared = tuple(sorted(principals_by_domain.values()))
    observed = ("probe_migrator",)
    grantees = (PUBLIC_PRINCIPAL, *declared, *observed)
    rows: list[ModelApplicationDatabaseAclRow] = []
    for principal in grantees:
        rows.append(
            ModelApplicationDatabaseAclRow(
                principal=principal,
                database_ref=database_ref,
                physical_database=physical_database,
                object_type=EnumDatabaseGrantObjectType.DATABASE,
                privileges=(EnumDatabasePrivilege.CONNECT,)
                if principal in declared
                else (),
            )
        )
        for schema_name, domain in schema_domains.items():
            rows.append(
                ModelApplicationDatabaseAclRow(
                    principal=principal,
                    database_ref=database_ref,
                    physical_database=physical_database,
                    object_type=EnumDatabaseGrantObjectType.SCHEMA,
                    schema_ref=schema_name,
                    privileges=(EnumDatabasePrivilege.USAGE,)
                    if principals_by_domain.get(domain) == principal
                    else (),
                )
            )
    defaults = tuple(
        ModelApplicationDatabaseDefaultAclRow(
            owner=owners[schema_name],
            database_ref=database_ref,
            physical_database=physical_database,
            schema_ref=schema_name,
            object_type=object_type,
            grantee=grantee,
        )
        for schema_name in sorted(schema_domains)
        for object_type in (
            EnumDatabaseGrantObjectType.TABLE,
            EnumDatabaseGrantObjectType.SEQUENCE,
            EnumDatabaseGrantObjectType.FUNCTION,
            EnumDatabaseGrantObjectType.TYPE,
        )
        for grantee in grantees
    )
    payload = {
        "authorization_scope": (
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
        "scaffold_status": "READY",
        "scaffold_blockers": (),
        "status": "BLOCKED",
        "blockers": (
            "scaffold proof deliberately withholds materialized object evidence",
        ),
        "sources": source_matrix.sources,
        "declared_principals": {database_ref: declared},
        "observed_principals": {database_ref: observed},
        "absent_principals": {database_ref: declared},
        "observed_owner_roles": (),
        "absent_owner_roles": tuple(sorted(owners.values())),
        "observed_role_states": (
            ModelApplicationDatabaseObservedRoleState(
                role="probe_migrator",
                login=False,
                superuser=False,
                bypass_rls=False,
                create_database=False,
                create_role=False,
                replication=False,
                inherit=False,
            ),
        ),
        "governed_role_states": (
            *(
                ModelApplicationDatabaseRoleState(
                    role=owner,
                    role_kind="owner",
                    login=False,
                )
                for owner in sorted(owners.values())
            ),
            *(
                ModelApplicationDatabaseRoleState(
                    role=principal,
                    role_kind="workload",
                    login=True,
                )
                for principal in declared
            ),
            ModelApplicationDatabaseRoleState(
                role="probe_migrator",
                role_kind="migration",
                login=False,
            ),
        ),
        "retained_administrative_principals": (),
        "database_owners": {database_ref: owners["platform_catalog"]},
        "required_connect_databases": (physical_database,),
        "observed_connect_database_owners": {physical_database: "postgres"},
        "allowed_connect_principals": {physical_database: declared},
        "observed_connect_principals": {physical_database: observed},
        "absent_connect_principals": {physical_database: declared},
        "schema_domains": {database_ref: schema_domains},
        "observed_schema_owners": {database_ref: {}},
        "absent_schemas": {database_ref: tuple(sorted(schema_domains))},
        "principal_domains": {
            principal: (domain,) for domain, principal in principals_by_domain.items()
        },
        "allowed_memberships": tuple(
            ModelApplicationDatabaseRoleMembership(
                database_ref=database_ref,
                role=owner,
                member="probe_migrator",
            )
            for owner in sorted(owners.values())
        ),
        "observed_objects": (),
        "objects": (),
        "rows": tuple(rows),
        "default_privileges": defaults,
        "excluded_objects": (),
    }
    return ModelApplicationDatabaseAclMatrix.model_validate(payload)


def _scaffold_phase_proof(matrix: ModelApplicationDatabaseAclMatrix) -> None:
    """Prove P1 can precede object materialization without weakening rollback."""
    probe = _scaffold_probe_matrix(matrix)
    probe_dsn = _admin_dsn_for_database("acl_scaffold_probe")
    assert probe.status == "BLOCKED"
    assert probe.scaffold_status == "READY"
    rendered = render_application_database_acl_sql(
        probe,
        allow_synthetic_proof=True,
        phase=EnumApplicationDatabaseAclRenderPhase.SCAFFOLD,
    )
    assert "-- Render phase: scaffold" in rendered
    assert "ALTER TABLE " not in rendered
    assert "ALTER VIEW " not in rendered
    assert "ALTER MATERIALIZED VIEW " not in rendered
    assert "ALTER FUNCTION " not in rendered
    assert "ALTER PROCEDURE " not in rendered
    assert "ALTER TYPE " not in rendered
    assert "GRANT SELECT ON TABLE " not in rendered

    prechange = _capture_snapshot(probe, dsn=probe_dsn)
    _legacy_scaffold_behavior_proof()
    application_prechange = _capture_snapshot(matrix)
    wrong_database = subprocess.run(
        ["psql", "--no-psqlrc", "--dbname", ADMIN_DSN],
        input=rendered,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_database.returncode != 0
    assert _capture_snapshot(probe, dsn=probe_dsn) == prechange
    assert _capture_snapshot(matrix) == application_prechange
    print("acl_phase=scaffold_wrong_database_guard status=PASS")

    collision_connection = _admin(probe_dsn)
    try:
        with collision_connection.cursor() as cursor:
            cursor.execute(
                "CREATE ROLE probe_catalog_reader LOGIN PASSWORD %s CREATEROLE",
                (ROLE_PASSWORD,),
            )
        hostile_role_snapshot = _capture_snapshot(probe, dsn=probe_dsn)
        hostile_role_apply = subprocess.run(
            ["psql", "--no-psqlrc", "--dbname", probe_dsn],
            input=rendered,
            text=True,
            capture_output=True,
            check=False,
        )
        assert hostile_role_apply.returncode != 0
        assert _capture_snapshot(probe, dsn=probe_dsn) == hostile_role_snapshot, (
            "stale expected-absent role evidence leaked a mutation"
        )
        with collision_connection.cursor() as cursor:
            cursor.execute("DROP ROLE probe_catalog_reader")
        assert _capture_snapshot(probe, dsn=probe_dsn) == prechange

        with collision_connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA tenant")
            cursor.execute("CREATE TABLE tenant.hostile_collision(id integer)")
        hostile_schema_snapshot = _capture_snapshot(probe, dsn=probe_dsn)
        hostile_schema_apply = subprocess.run(
            ["psql", "--no-psqlrc", "--dbname", probe_dsn],
            input=rendered,
            text=True,
            capture_output=True,
            check=False,
        )
        assert hostile_schema_apply.returncode != 0
        assert _capture_snapshot(probe, dsn=probe_dsn) == hostile_schema_snapshot, (
            "stale expected-absent schema evidence leaked a mutation"
        )
        with collision_connection.cursor() as cursor:
            cursor.execute("DROP SCHEMA tenant CASCADE")
        assert _capture_snapshot(probe, dsn=probe_dsn) == prechange
    finally:
        collision_connection.close()
    print("acl_phase=scaffold_stale_absence_collision status=REJECTED")

    _atomic_failure_proof(
        probe,
        prechange,
        dsn=probe_dsn,
        phase=EnumApplicationDatabaseAclRenderPhase.SCAFFOLD,
    )
    for pass_number in (1, 2):
        _execute_matrix(
            probe,
            dsn=probe_dsn,
            phase=EnumApplicationDatabaseAclRenderPhase.SCAFFOLD,
        )
        print(f"acl_phase=scaffold_apply pass={pass_number} status=PASS")
    green = _capture_snapshot(probe, dsn=probe_dsn)
    expected_additions = _expected_acl_set(probe)
    actual_acl = _actual_acl_set(green, "acl_scaffold_probe")
    assert expected_additions <= actual_acl, expected_additions - actual_acl
    assert green["database_owner"] == prechange["database_owner"]
    expected_schema_owners = {
        (schema_name, owner)
        for schema_name, owner in {
            row.schema_ref: row.owner for row in probe.default_privileges
        }.items()
    }
    assert expected_schema_owners <= {
        (str(row["schema_name"]), str(row["owner"])) for row in green["schema_owners"]
    }
    for role in green["roles"]:
        if str(role["rolname"]) == "probe_migrator":
            continue
        assert not any(
            bool(role[field])
            for field in (
                "rolsuper",
                "rolinherit",
                "rolcreaterole",
                "rolcreatedb",
                "rolreplication",
                "rolbypassrls",
            )
        ), role
    assert not green["object_owners"]
    assert not green["object_acl"]
    assert not green["column_acl"]
    assert not green["default_acl"]
    assert green[
        "default_acl_catalog_rows"
    ] == _expected_hardened_default_acl_catalog_rows(probe)
    _legacy_scaffold_behavior_proof()

    hostile_connection = _admin(probe_dsn)
    try:
        for create_statement, drop_statement in (
            (
                'CREATE COLLATION tenant.hostile_collation FROM "C"',
                "DROP COLLATION tenant.hostile_collation",
            ),
            (
                "CREATE TYPE tenant.hostile_enum AS ENUM ('collision')",
                "DROP TYPE tenant.hostile_enum",
            ),
        ):
            with hostile_connection.cursor() as cursor:
                cursor.execute(create_statement)
            hostile_snapshot = _capture_snapshot(probe, dsn=probe_dsn)
            hostile_apply = subprocess.run(
                ["psql", "--no-psqlrc", "--dbname", probe_dsn],
                input=rendered,
                text=True,
                capture_output=True,
                check=False,
            )
            assert hostile_apply.returncode != 0
            assert "expected-absent schema collision for tenant" in hostile_apply.stderr
            assert _capture_snapshot(probe, dsn=probe_dsn) == hostile_snapshot
            with hostile_connection.cursor() as cursor:
                cursor.execute(drop_statement)
        assert _capture_snapshot(probe, dsn=probe_dsn) == green
    finally:
        hostile_connection.close()

    full_payload = probe.model_dump(mode="json")
    full_payload.update({"status": "READY", "blockers": []})
    full_probe = ModelApplicationDatabaseAclMatrix.model_validate(full_payload)
    for pass_number in (1, 2):
        _execute_matrix(full_probe, dsn=probe_dsn)
        print(f"acl_phase=scaffold_probe_full_apply pass={pass_number} status=PASS")
    full_green = _capture_snapshot(full_probe, dsn=probe_dsn)
    assert not _catalog_violations(full_green, full_probe), _catalog_violations(
        full_green,
        full_probe,
    )
    _expect_connection_denied("legacy_probe_login", "acl_scaffold_probe")
    _restore_snapshot(prechange, probe, dsn=probe_dsn)
    assert _capture_snapshot(probe, dsn=probe_dsn) == prechange
    _legacy_scaffold_behavior_proof()
    print("acl_phase=scaffold_fresh_additive_round_trip status=PASS")


def main() -> None:
    postgres_major = _postgres_major()
    assert postgres_major == 16, postgres_major
    matrix = _fixture_matrix()
    assert matrix.status == "READY", matrix.blockers
    assert matrix.scaffold_status == "BLOCKED"
    assert any(
        "additive scaffold intended roles are not already safe" in blocker
        for blocker in matrix.scaffold_blockers
    )

    _scaffold_phase_proof(matrix)
    _extra_object_rejection_proof(matrix)
    legacy_default_acl = _capture_legacy_default_acl()
    assert legacy_default_acl
    _legacy_default_behavior_proof()
    prechange = _capture_snapshot(matrix)
    OBSERVED_PRECHANGE.write_text(
        json.dumps(prechange, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update = os.environ.get("UPDATE_PRECHANGE", "false").lower() == "true"
    if update:
        rollback_input = prechange
    else:
        rollback_input = cast(
            "_AclSnapshot",
            json.loads(EXPECTED_PRECHANGE.read_text(encoding="utf-8")),
        )
        assert rollback_input == prechange, "durable pre-change ACL artifact drift"

    red = _catalog_violations(prechange, matrix)
    required_red = {
        "OWNER_MISMATCH",
        "PUBLIC_PRIVILEGE",
        "BROAD_GRANT",
        "UNSAFE_DEFAULT_PRIVILEGE",
        "CROSS_DOMAIN_PRIVILEGE",
        "RUNTIME_ROLE_ATTRIBUTES",
        "OWNER_MEMBERSHIP",
        "UNDECLARED_PRINCIPAL_PRIVILEGE",
        "MISSING_DECLARED_MEMBERSHIP",
        "RUNTIME_OWNERSHIP",
        "GRANT_OPTION",
        "COLUMN_PRIVILEGE",
        "DATABASE_DDL_PRIVILEGE",
    }
    assert required_red <= red, (required_red - red, red)
    print(f"acl_phase=prechange red_controls={','.join(sorted(red))} status=DETECTED")

    _atomic_failure_proof(matrix, prechange)
    for pass_number in (1, 2):
        _execute_matrix(matrix)
        print(f"acl_phase=apply pass={pass_number} status=PASS")
    green = _capture_snapshot(matrix)
    assert not _catalog_violations(green, matrix), _catalog_violations(green, matrix)
    assert [row for row in green["roles"] if row["rolname"] == "rls_admin"] == [
        row for row in prechange["roles"] if row["rolname"] == "rls_admin"
    ]
    assert [
        row for row in green["database_owner"] if row["database_name"] != DATABASE
    ] == [
        row for row in prechange["database_owner"] if row["database_name"] != DATABASE
    ]
    assert _capture_legacy_default_acl() == legacy_default_acl
    _legacy_default_behavior_proof()
    _unknown_acl_and_membership_reconciliation_proof(matrix, green)
    try:
        _restore_snapshot(rollback_input, matrix, inject_failure=True)
    except psycopg2.errors.DivisionByZero:
        pass
    else:
        raise AssertionError("injected rollback failure unexpectedly succeeded")
    assert _capture_snapshot(matrix) == green, (
        "failed rollback leaked a partial catalog mutation"
    )
    print("acl_phase=rollback_atomic_failure red_control=ROLLBACK status=PASS")
    _behavioral_proof(matrix)
    _migration_membership_proof()
    _future_default_proof()
    print("acl_phase=postgres16_readback status=PASS")

    _restore_snapshot(rollback_input, matrix)
    restored = _capture_snapshot(matrix)
    assert restored == prechange, "rollback did not reproduce durable pre-change ACL"
    assert _capture_legacy_default_acl() == legacy_default_acl
    _legacy_default_behavior_proof()
    _builtin_default_rollback_proof()
    print("acl_phase=rollback_round_trip status=PASS")

    for pass_number in (1, 2):
        _execute_matrix(matrix)
        print(f"acl_phase=reapply pass={pass_number} status=PASS")
    final = _capture_snapshot(matrix)
    assert not _catalog_violations(final, matrix), _catalog_violations(final, matrix)
    assert _capture_legacy_default_acl() == legacy_default_acl
    _legacy_default_behavior_proof()
    print(
        f"acl_status=PASS postgres_major={postgres_major} objects={len(matrix.objects)} "
        f"rows={len(matrix.rows)} defaults={len(matrix.default_privileges)}"
    )


if __name__ == "__main__":
    main()
