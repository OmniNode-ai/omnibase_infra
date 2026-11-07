# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration grant proofs for the view seam readings table (OMN-22020)."""

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
    / "node_view_seam_reading_reducer"
)
CREATE_FILE = MIGRATION_DIR / "0001_create_view_seam_readings.sql"
GRANT_FILE = (
    MIGRATION_DIR / "0002_grant_tenant_projection_writer_view_seam_readings.sql"
)


@pytest.mark.integration
def test_grant_text_allows_exactly_select_insert_update() -> None:
    sql_text = re.sub(r"--[^\n]*|/\*.*?\*/", "", GRANT_FILE.read_text(), flags=re.S)
    grants = re.findall(
        r"\bGRANT\s+([^;]+?)\s+ON\s+(?:TABLE\s+)?"
        r"public\.view_seam_readings\s+TO\s+tenant_projection_writer\s*;",
        sql_text,
        flags=re.I,
    )
    assert len(grants) == 1, grants
    privileges = [privilege.strip().upper() for privilege in grants[0].split(",")]
    assert sorted(privileges) == ["INSERT", "SELECT", "UPDATE"]
    assert re.search(r"\b(?:DELETE|TRUNCATE)\b", sql_text, flags=re.I) is None


@pytest.mark.integration
def test_migrations_grant_writer_privileges(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    setup = ephemeral_postgres.connect()
    try:
        setup.autocommit = True
        with setup.cursor() as cur:
            cur.execute("CREATE ROLE tenant_projection_writer NOLOGIN NOBYPASSRLS")
    finally:
        setup.close()
    for migration in (CREATE_FILE, GRANT_FILE):
        result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert result.returncode == 0, f"{migration.name}: {result.stderr}"

    verify = ephemeral_postgres.connect()
    try:
        with verify.cursor() as cur:
            for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                cur.execute(
                    "SELECT has_table_privilege('tenant_projection_writer', "
                    "'public.view_seam_readings', %s)",
                    (privilege,),
                )
                assert cur.fetchone() == (privilege != "DELETE",), privilege
    finally:
        verify.close()


@pytest.mark.integration
def test_grant_migration_refuses_missing_writer_role(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    created = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(CREATE_FILE))
    assert created.returncode == 0, created.stderr
    refused = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(GRANT_FILE))
    assert refused.returncode != 0, refused.stderr
    assert "tenant_projection_writer role missing" in refused.stderr, refused.stderr
