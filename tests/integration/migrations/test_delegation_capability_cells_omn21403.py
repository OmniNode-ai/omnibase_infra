# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration grant and RLS proofs for delegation capability cells (OMN-21403)."""

from __future__ import annotations

import re
from pathlib import Path

import psycopg2.errors
import pytest
from psycopg2 import sql

from tests.integration.migrations.conftest import EphemeralPostgres

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATION_DIR = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_delegation_capability_map"
)
CREATE_FILE = MIGRATION_DIR / "0000_create_delegation_capability_cells.sql"
GRANT_FILE = (
    MIGRATION_DIR
    / "0001_grant_tenant_projection_writer_delegation_capability_cells.sql"
)
FORCE_RLS_FILE = MIGRATION_DIR / "0002_force_rls_delegation_capability_cells.sql"
TENANT_A = "11111111-1111-1111-1111-111111111111"
TENANT_B = "22222222-2222-2222-2222-222222222222"
INSERT_CELL = """
    INSERT INTO public.delegation_capability_cells VALUES (
        %s::uuid, %s, 'model', 'backend', 'config', 'code', 'S', '{}'::jsonb,
        2, 1, 0.5, 0.1, 0.9, 0.1, 0.5, 2, 'MISSED', NOW()
    )
"""


def _apply_migrations(server: EphemeralPostgres) -> None:
    setup = server.connect()
    try:
        setup.autocommit = True
        with setup.cursor() as cur:
            cur.execute("CREATE ROLE tenant_projection_writer NOLOGIN NOBYPASSRLS")
            cur.execute("CREATE ROLE app_dashboard NOLOGIN NOBYPASSRLS")
    finally:
        setup.close()
    for migration in (CREATE_FILE, GRANT_FILE, FORCE_RLS_FILE):
        result = server.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, f"{migration.name}: {result.stderr}"


def _seed_tenants(server: EphemeralPostgres) -> None:
    setup = server.connect()
    try:
        setup.autocommit = True
        with setup.cursor() as cur:
            cur.execute(INSERT_CELL, (TENANT_A, "run-a"))
            cur.execute(INSERT_CELL, (TENANT_B, "run-b"))
    finally:
        setup.close()


@pytest.mark.integration
def test_grant_text_allows_exactly_select_insert_update() -> None:
    sql = re.sub(r"--[^\n]*|/\*.*?\*/", "", GRANT_FILE.read_text(), flags=re.S)
    grants = re.findall(
        r"\bGRANT\s+([^;]+?)\s+ON\s+(?:TABLE\s+)?"
        r"public\.delegation_capability_cells\s+TO\s+tenant_projection_writer\s*;",
        sql,
        flags=re.I,
    )
    assert len(grants) == 1, grants
    privileges = [privilege.strip().upper() for privilege in grants[0].split(",")]
    assert sorted(privileges) == ["INSERT", "SELECT", "UPDATE"]
    assert re.search(r"\b(?:DELETE|TRUNCATE)\b", sql, flags=re.I) is None


@pytest.mark.integration
def test_migrations_grant_writer_privileges_and_force_rls(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply_migrations(ephemeral_postgres)

    verify = ephemeral_postgres.connect()
    try:
        with verify.cursor() as cur:
            for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                cur.execute(
                    "SELECT has_table_privilege('tenant_projection_writer', "
                    "'public.delegation_capability_cells', %s)",
                    (privilege,),
                )
                assert cur.fetchone() == (privilege != "DELETE",), privilege
            cur.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE oid = 'public.delegation_capability_cells'::regclass"
            )
            assert cur.fetchone() == (True, True)
    finally:
        verify.close()


@pytest.mark.integration
def test_grant_migration_refuses_missing_writer_role(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    setup = ephemeral_postgres.connect()
    try:
        setup.autocommit = True
        with setup.cursor() as cur:
            cur.execute("CREATE ROLE app_dashboard NOLOGIN")
    finally:
        setup.close()

    created = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(CREATE_FILE))
    assert created.returncode == 0, created.stderr
    refused = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(GRANT_FILE))
    assert refused.returncode != 0, refused.stderr
    assert "tenant_projection_writer role missing" in refused.stderr, refused.stderr


@pytest.mark.integration
def test_writer_operations_enforce_tenant_isolation(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply_migrations(ephemeral_postgres)
    _seed_tenants(ephemeral_postgres)
    writer = ephemeral_postgres.connect()
    try:
        writer.autocommit = True
        with writer.cursor() as cur:
            cur.execute("SET ROLE tenant_projection_writer")
            cur.execute("SELECT current_user")
            assert cur.fetchone() == ("tenant_projection_writer",)
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_A,))
            cur.execute(INSERT_CELL, (TENANT_A, "writer-run"))
            cur.execute(
                "UPDATE public.delegation_capability_cells SET passes = 2 "
                "WHERE map_run_id = 'writer-run' RETURNING passes"
            )
            assert cur.fetchall() == [(2,)]
            cur.execute(
                "SELECT map_run_id FROM public.delegation_capability_cells "
                "ORDER BY map_run_id"
            )
            assert cur.fetchall() == [("run-a",), ("writer-run",)]
            cur.execute(
                "UPDATE public.delegation_capability_cells SET passes = 0 "
                "WHERE tenant_id = %s::uuid",
                (TENANT_B,),
            )
            assert cur.rowcount == 0
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(INSERT_CELL, (TENANT_B, "cross-tenant"))
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(
                    "UPDATE public.delegation_capability_cells SET tenant_id = %s::uuid "
                    "WHERE map_run_id = 'writer-run'",
                    (TENANT_B,),
                )
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("DELETE FROM public.delegation_capability_cells")
    finally:
        writer.close()


@pytest.mark.integration
@pytest.mark.parametrize("role", ["tenant_projection_writer", "app_dashboard"])
def test_unset_tenant_context_denies_all_rows(
    ephemeral_postgres: EphemeralPostgres,
    role: str,
) -> None:
    _apply_migrations(ephemeral_postgres)
    _seed_tenants(ephemeral_postgres)
    reader = ephemeral_postgres.connect()
    try:
        reader.autocommit = True
        with reader.cursor() as cur:
            # The role is selected only from the two fixed test parameters.
            cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
            cur.execute("SELECT current_setting('app.tenant_id', true)")
            assert cur.fetchone() == (None,)
            cur.execute("SELECT count(*) FROM public.delegation_capability_cells")
            assert cur.fetchone() == (0,)
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute(INSERT_CELL, (TENANT_A, "without-context"))
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_A,))
            cur.execute("SELECT map_run_id FROM public.delegation_capability_cells")
            assert cur.fetchall() == [("run-a",)]
            for invalid_tenant in ("", "invalid-uuid"):
                cur.execute(
                    "SELECT set_config('app.tenant_id', %s, false)", (invalid_tenant,)
                )
                with pytest.raises(psycopg2.errors.InvalidTextRepresentation):
                    cur.execute("SELECT * FROM public.delegation_capability_cells")
    finally:
        reader.close()


@pytest.mark.integration
def test_reconciliation_refuses_unknown_required_values_without_losing_rows(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    seeded = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE TABLE public.delegation_capability_cells (map_run_id TEXT); "
        "INSERT INTO public.delegation_capability_cells VALUES ('keep-me');",
    )
    assert seeded.returncode == 0, seeded.stderr
    refused = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(CREATE_FILE))
    assert refused.returncode != 0
    assert 'column "tenant_id"' in refused.stderr
    assert "contains null values" in refused.stderr
    verify = ephemeral_postgres.connect()
    try:
        with verify.cursor() as cur:
            cur.execute(
                "SELECT tenant_id, map_run_id FROM public.delegation_capability_cells"
            )
            assert cur.fetchall() == [(None, "keep-me")]
    finally:
        verify.close()


@pytest.mark.integration
def test_reconciliation_restores_checks_on_an_existing_table(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply_migrations(ephemeral_postgres)
    setup = ephemeral_postgres.connect()
    try:
        setup.autocommit = True
        with setup.cursor() as cur:
            for suffix in ("size_band", "n", "passes", "verdict"):
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE public.delegation_capability_cells DROP CONSTRAINT {}"
                    ).format(
                        sql.Identifier(f"delegation_capability_cells_{suffix}_check")
                    )
                )
        restored = ephemeral_postgres.psql(
            "-v", "ON_ERROR_STOP=1", "-f", str(CREATE_FILE)
        )
        assert restored.returncode == 0, restored.stderr
        with setup.cursor() as cur:
            for column, value in (
                ("size_band", "X"),
                ("n", 0),
                ("passes", 3),
                ("verdict", "INVALID"),
            ):
                cur.execute(INSERT_CELL, (TENANT_A, "constraint-run"))
                with pytest.raises(psycopg2.errors.CheckViolation):
                    cur.execute(
                        sql.SQL(
                            "UPDATE public.delegation_capability_cells SET {} = %s"
                        ).format(sql.Identifier(column)),
                        (value,),
                    )
                cur.execute("DELETE FROM public.delegation_capability_cells")
    finally:
        setup.close()


@pytest.mark.integration
def test_dashboard_grants_follow_flat_role_provisioning(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    role_migration = (
        REPO_ROOT / "docker/migrations/forward/094_create_app_dashboard_role.sql"
    )
    for migration in (role_migration, CREATE_FILE, FORCE_RLS_FILE):
        result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, f"{migration.name}: {result.stderr}"
    _seed_tenants(ephemeral_postgres)
    verify = ephemeral_postgres.connect()
    try:
        with verify.cursor() as cur:
            cur.execute("SET ROLE app_dashboard")
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_A,))
            cur.execute("SELECT map_run_id FROM public.delegation_capability_cells")
            assert cur.fetchall() == [("run-a",)]
    finally:
        verify.close()


@pytest.mark.integration
def test_reapplying_migrations_preserves_cells_and_isolation(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply_migrations(ephemeral_postgres)
    _seed_tenants(ephemeral_postgres)
    for migration in (CREATE_FILE, GRANT_FILE, FORCE_RLS_FILE):
        result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, f"{migration.name}: {result.stderr}"
    verify = ephemeral_postgres.connect()
    try:
        with verify.cursor() as cur:
            cur.execute(
                "SELECT map_run_id FROM public.delegation_capability_cells ORDER BY map_run_id"
            )
            assert cur.fetchall() == [("run-a",), ("run-b",)]
            cur.execute("SET ROLE tenant_projection_writer")
            cur.execute("SELECT set_config('app.tenant_id', %s, false)", (TENANT_B,))
            cur.execute("SELECT map_run_id FROM public.delegation_capability_cells")
            assert cur.fetchall() == [("run-b",)]
    finally:
        verify.close()
