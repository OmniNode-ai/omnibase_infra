# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration grant and RLS proofs for delegation capability cells (OMN-21403)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
    setup = ephemeral_postgres.connect()
    try:
        setup.autocommit = True
        with setup.cursor() as cur:
            cur.execute("CREATE ROLE tenant_projection_writer NOLOGIN")
            cur.execute("CREATE ROLE app_dashboard NOLOGIN")
    finally:
        setup.close()

    for migration in (CREATE_FILE, GRANT_FILE, FORCE_RLS_FILE):
        result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, f"{migration.name}: {result.stderr}"

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
