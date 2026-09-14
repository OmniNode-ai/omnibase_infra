#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Apply the generated application-database ACL matrix to a live database.

OMN-15355. This is the ONLY sanctioned way to put the generated GRANT/REVOKE
program onto a live PostgreSQL instance. Hand-run ``psql`` is not a path, and
neither is a hand-written approximation of the rollback.

Usage::

    ADMIN_DSN=... \\
    ACL_ROLE_DSN_APP_DASHBOARD=... \\
    uv run python scripts/apply_application_database_acl.py \\
        --matrix docker/application-acl-proof/generated/candidate-matrix.yaml \\
        --ticket OMN-15355 \\
        --consent-citation docs/tracking/ROLLING_WORK_LEDGER.md:7789 \\
        --ledger-root "$OMNI_HOME/omni_home" \\
        --snapshot-out evidence/prechange-acl.json \\
        --execute

Without ``--execute`` the run is a dry run: it resolves consent, refuses an
ineligible matrix, writes the durable pre-change snapshot, and prints the probe
plan -- but mutates nothing.

No credential value is read into a report, a log line, or a traceback. Each
per-principal probe DSN is read from its own environment variable, named by
``resolve_role_dsn_env_name``, and the value never leaves ``psycopg2``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import psycopg2
import psycopg2.extras
import yaml
from psycopg2 import sql

from omnibase_infra.validation.application_database_acl_apply import (
    AclApplyRefusalError,
    apply_application_database_acl,
)
from omnibase_infra.validation.enums.enum_acl_connection_probe_kind import (
    EnumAclConnectionProbeKind,
)
from omnibase_infra.validation.enums.enum_application_database_acl_render_phase import (
    EnumApplicationDatabaseAclRenderPhase,
)
from omnibase_infra.validation.models.model_acl_connection_probe import (
    ModelAclConnectionProbe,
    resolve_role_dsn_env_name,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
)

_ROLE_ATTRIBUTE_QUERY = """
SELECT rolname, rolcanlogin, rolsuper, rolinherit, rolcreaterole,
       rolcreatedb, rolreplication, rolbypassrls
FROM pg_roles
WHERE rolname = ANY(%s)
ORDER BY rolname
"""

_MEMBERSHIP_QUERY = """
SELECT parent.rolname AS parent_role, member.rolname AS member_role,
       grantor.rolname AS grantor, membership.admin_option
FROM pg_auth_members membership
JOIN pg_roles parent ON parent.oid = membership.roleid
JOIN pg_roles member ON member.oid = membership.member
JOIN pg_roles grantor ON grantor.oid = membership.grantor
WHERE member.rolname = ANY(%s) OR parent.rolname = ANY(%s)
ORDER BY parent_role, member_role
"""

_DATABASE_ACL_QUERY = """
SELECT database.datname AS database_name,
       owner.rolname AS owner,
       COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
       grantor.rolname AS grantor,
       acl.privilege_type,
       acl.is_grantable
FROM pg_database database
JOIN pg_roles owner ON owner.oid = database.datdba
CROSS JOIN LATERAL aclexplode(
    COALESCE(database.datacl, acldefault('d', database.datdba))
) acl
LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
JOIN pg_roles grantor ON grantor.oid = acl.grantor
WHERE database.datname = ANY(%s)
ORDER BY database_name, grantee, privilege_type
"""

_SCHEMA_ACL_QUERY = """
SELECT namespace.nspname AS schema_name,
       owner.rolname AS owner,
       COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
       acl.privilege_type,
       acl.is_grantable
FROM pg_namespace namespace
JOIN pg_roles owner ON owner.oid = namespace.nspowner
CROSS JOIN LATERAL aclexplode(
    COALESCE(namespace.nspacl, acldefault('n', namespace.nspowner))
) acl
LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
WHERE namespace.nspname NOT LIKE 'pg\\_%%' AND namespace.nspname <> 'information_schema'
ORDER BY schema_name, grantee, privilege_type
"""

_RELATION_ACL_QUERY = """
SELECT namespace.nspname AS schema_name,
       relation.relname AS relation_name,
       relation.relkind,
       owner.rolname AS owner,
       COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
       acl.privilege_type,
       acl.is_grantable
FROM pg_class relation
JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
JOIN pg_roles owner ON owner.oid = relation.relowner
CROSS JOIN LATERAL aclexplode(
    COALESCE(relation.relacl, acldefault('r', relation.relowner))
) acl
LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
WHERE namespace.nspname NOT LIKE 'pg\\_%%'
  AND namespace.nspname <> 'information_schema'
  AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
ORDER BY schema_name, relation_name, grantee, privilege_type
"""

_DEFAULT_ACL_QUERY = """
SELECT owner.rolname AS owner,
       COALESCE(namespace.nspname, '') AS schema_name,
       default_acl.defaclobjtype AS object_type,
       COALESCE(grantee.rolname, 'PUBLIC') AS grantee,
       acl.privilege_type,
       acl.is_grantable
FROM pg_default_acl default_acl
JOIN pg_roles owner ON owner.oid = default_acl.defaclrole
LEFT JOIN pg_namespace namespace ON namespace.oid = default_acl.defaclnamespace
CROSS JOIN LATERAL aclexplode(default_acl.defaclacl) acl
LEFT JOIN pg_roles grantee ON grantee.oid = acl.grantee
ORDER BY owner, schema_name, object_type, grantee, privilege_type
"""


def _rows(
    dsn: str, statement: str, parameters: Sequence[object] = ()
) -> list[dict[str, object]]:
    with psycopg2.connect(dsn) as connection:
        with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(statement, parameters)
            return [dict(row) for row in cursor.fetchall()]


def _controlled_roles(matrix: ModelApplicationDatabaseAclMatrix) -> list[str]:
    roles: set[str] = set(matrix.observed_owner_roles) | set(matrix.absent_owner_roles)
    for principals in matrix.declared_principals.values():
        roles.update(principals)
    for principals in matrix.allowed_connect_principals.values():
        roles.update(principals)
    for principals in matrix.observed_connect_principals.values():
        roles.update(principals)
    return sorted(roles)


def _capture_live_snapshot(
    matrix: ModelApplicationDatabaseAclMatrix, *, admin_dsn: str
) -> dict[str, object]:
    """Read the complete pre-change ACL state that rollback is driven from.

    Every ACL surface the rendered program can touch is captured, including the
    ones it does not currently touch: a snapshot that records only what the
    change is expected to alter cannot prove the change altered nothing else.
    """
    roles = _controlled_roles(matrix)
    databases = list(matrix.required_connect_databases)
    return {
        "schema_version": "1.0",
        "captured_for_ticket": "OMN-15355",
        "required_connect_databases": databases,
        "controlled_roles": roles,
        "role_attributes": _rows(admin_dsn, _ROLE_ATTRIBUTE_QUERY, (roles,)),
        "role_memberships": _rows(admin_dsn, _MEMBERSHIP_QUERY, (roles, roles)),
        "database_acl": _rows(admin_dsn, _DATABASE_ACL_QUERY, (databases,)),
        "schema_acl": _rows(admin_dsn, _SCHEMA_ACL_QUERY),
        "relation_acl": _rows(admin_dsn, _RELATION_ACL_QUERY),
        "default_acl": _rows(admin_dsn, _DEFAULT_ACL_QUERY),
    }


def _execute_sql(rendered: str, *, admin_dsn: str) -> None:
    """Apply the rendered program through psql; it carries its own transaction."""
    result = subprocess.run(
        ["psql", "--no-psqlrc", "--variable", "ON_ERROR_STOP=1", "--dbname", admin_dsn],
        input=rendered,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AclApplyRefusalError(
            "the rendered ACL program failed to apply; it is transactional, so "
            "nothing was committed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_probe(probe: ModelAclConnectionProbe, *, admin_dsn: str) -> bool:
    """Open one connection and report whether the outcome matches the assertion.

    A probe whose DSN is not supplied raises ``LookupError``, which the apply
    path turns into a refusal. That is deliberate: a probe that could not run
    reads exactly like a probe that passed, and the second one is a lie.
    """
    if probe.kind is EnumAclConnectionProbeKind.NEGATIVE_PUBLIC:
        return _public_connect_revoked(probe.database, admin_dsn=admin_dsn)
    assert probe.principal is not None
    env_name = resolve_role_dsn_env_name(probe.principal)
    dsn = os.environ.get(env_name)
    if not dsn:
        raise LookupError(f"{env_name} is not set")
    try:
        connection = psycopg2.connect(dsn, dbname=probe.database, connect_timeout=10)
    except psycopg2.OperationalError:
        return probe.kind is EnumAclConnectionProbeKind.NEGATIVE
    connection.close()
    return probe.kind is EnumAclConnectionProbeKind.POSITIVE


def _public_connect_revoked(database: str, *, admin_dsn: str) -> bool:
    """Assert PUBLIC holds no CONNECT on this database, implicit ACL included."""
    rows = _rows(
        admin_dsn,
        """
        SELECT COUNT(*) AS granted
        FROM pg_database database
        CROSS JOIN LATERAL aclexplode(
            COALESCE(database.datacl, acldefault('d', database.datdba))
        ) acl
        WHERE database.datname = %s
          AND acl.grantee = 0
          AND acl.privilege_type = 'CONNECT'
        """,
        (database,),
    )
    granted = rows[0]["granted"]
    assert isinstance(granted, int)
    return granted == 0


def _restore_from_snapshot(snapshot_path: Path, *, admin_dsn: str) -> None:
    """Rebuild the recorded database-level ACLs from the durable snapshot."""
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    statements: list[sql.Composable] = []
    by_database: dict[str, list[Mapping[str, str]]] = {}
    for row in snapshot.get("database_acl", []):
        by_database.setdefault(str(row["database_name"]), []).append(row)
    for database, rows in sorted(by_database.items()):
        statements.append(
            sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(
                sql.Identifier(database)
            )
        )
        for grantee in sorted(
            {row["grantee"] for row in rows if row["grantee"] != "PUBLIC"}
        ):
            statements.append(
                sql.SQL("REVOKE ALL ON DATABASE {} FROM {}").format(
                    sql.Identifier(database), sql.Identifier(grantee)
                )
            )
        for row in rows:
            target = (
                sql.SQL("PUBLIC")
                if row["grantee"] == "PUBLIC"
                else sql.Identifier(row["grantee"])
            )
            statements.append(
                sql.SQL("GRANT {} ON DATABASE {} TO {}").format(
                    sql.SQL(row["privilege_type"]),
                    sql.Identifier(database),
                    target,
                )
            )
    with psycopg2.connect(admin_dsn) as connection:
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
    print(
        f"acl_apply restore=DONE source={snapshot_path} statements={len(statements)}",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--consent-citation", required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--snapshot-out", type=Path, required=True)
    parser.add_argument(
        "--render-phase",
        type=EnumApplicationDatabaseAclRenderPhase,
        choices=tuple(EnumApplicationDatabaseAclRenderPhase),
        default=EnumApplicationDatabaseAclRenderPhase.FULL,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Mutate the live database; without it the run is a dry run",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    admin_dsn = os.environ.get("ADMIN_DSN")
    if not admin_dsn:
        print("acl_apply status=REFUSED reason=ADMIN_DSN_absent", file=sys.stderr)
        return 2
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(
        yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    )
    try:
        report = apply_application_database_acl(
            matrix,
            consent_citation=args.consent_citation,
            ticket=args.ticket,
            ledger_root=args.ledger_root,
            snapshot_path=args.snapshot_out,
            capture_snapshot=lambda m: _capture_live_snapshot(m, admin_dsn=admin_dsn),
            execute_sql=lambda rendered: _execute_sql(rendered, admin_dsn=admin_dsn),
            run_probe=lambda probe: _run_probe(probe, admin_dsn=admin_dsn),
            restore_snapshot=lambda path: _restore_from_snapshot(
                path, admin_dsn=admin_dsn
            ),
            phase=args.render_phase,
            execute=args.execute,
        )
    except AclApplyRefusalError as refusal:
        print(f"acl_apply status=REFUSED\n{refusal}", file=sys.stderr)
        return 1
    print(
        f"acl_apply status=PASS mutated={report.mutated} "
        f"probes_run={report.probes_run} probes_passed={report.probes_passed} "
        f"snapshot={report.snapshot_path} "
        f"consent={report.consent.citation} approved_by={report.consent.approved_by}"
    )
    for description in report.probe_descriptions:
        print(f"  probe {description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
