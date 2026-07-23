# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration contract checks for the app_dashboard role (OMN-14899).

The connecting role — not the RLS policy — is the real isolation control:
Postgres silently bypasses row-level security for SUPERUSER / BYPASSRLS
roles and for table owners. These checks pin the migration text to the
security-critical properties so a later edit cannot silently weaken them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATION_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "094_create_app_dashboard_role.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_094_create_app_dashboard_role.sql"
)


@pytest.mark.integration
def test_094_creates_role_with_bypass_flags_off() -> None:
    sql = MIGRATION_FILE.read_text()

    assert "CREATE ROLE app_dashboard" in sql
    assert "NOSUPERUSER" in sql
    assert "NOBYPASSRLS" in sql
    assert "NOLOGIN" in sql
    assert "NOCREATEDB" in sql
    assert "NOCREATEROLE" in sql


@pytest.mark.integration
def test_094_enforces_flags_on_preexisting_role() -> None:
    """ALTER ROLE must re-assert the flags — presence is not the property."""
    sql = MIGRATION_FILE.read_text()

    assert "ALTER ROLE app_dashboard" in sql
    # The ALTER block (after the guarded CREATE) must carry both
    # security-critical negations.
    alter_block = sql.split("ALTER ROLE app_dashboard", 1)[1]
    assert "NOSUPERUSER" in alter_block
    assert "NOBYPASSRLS" in alter_block


@pytest.mark.integration
def test_094_contains_no_credential_material() -> None:
    sql = MIGRATION_FILE.read_text()

    assert (
        "PASSWORD '" not in sql.upper() and "ENCRYPTED PASSWORD" not in sql.upper()
    ), (
        "credential material must never live in a migration; the LOGIN + "
        "password attach is a deployment-owned gated step"
    )


@pytest.mark.integration
def test_094_is_role_only_no_grants_no_connect() -> None:
    """Grants ride with the RLS migrations (OMN-14894), never with the role.

    Keeping the file free of \\connect and GRANT means it is valid in any
    database context, and a table can never become readable by app_dashboard
    before its tenant_isolation policy exists.
    """
    sql = MIGRATION_FILE.read_text()
    executable = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )

    assert "GRANT" not in executable.upper(), "role migration must not carry grants"
    assert "\\connect" not in executable
    assert "ALTER DEFAULT PRIVILEGES" not in executable.upper()


@pytest.mark.integration
def test_094_rollback_drops_role_and_grants() -> None:
    sql = ROLLBACK_FILE.read_text()

    assert "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM app_dashboard" in sql
    assert "DROP ROLE IF EXISTS app_dashboard" in sql
