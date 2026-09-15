# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ACL and non-destructive backout proof for the restricted claim principal."""

from __future__ import annotations

from pathlib import Path

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_ROOT = Path(__file__).resolve().parents[4]
_MIGRATION = (
    _ROOT / "docker/migrations/forward/107_create_action_authorization_nonce_claim.sql"
)
_ROLLBACK = (
    _ROOT
    / "docker/migrations/rollback/rollback_107_create_action_authorization_nonce_claim.sql"
)
_ROLE = "rsd_action_authorization_claim"


def _apply(ephemeral_postgres: EphemeralPostgres, migration: Path) -> None:
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
    assert result.returncode == 0, result.stderr


def _provision_claim_schema(ephemeral_postgres: EphemeralPostgres) -> None:
    result = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA action_authorization_claim",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_postgres16_restricted_principal_has_only_the_atomic_claim_function(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _provision_claim_schema(ephemeral_postgres)
    _apply(ephemeral_postgres, _MIGRATION)
    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT "
                "has_schema_privilege(%s, 'action_authorization_claim', 'USAGE'), "
                "has_table_privilege(%s, 'action_authorization_claim.nonce_claims', 'SELECT'), "
                "has_table_privilege(%s, 'action_authorization_claim.nonce_claims', 'INSERT'), "
                "pg_get_userbyid(c.relowner) <> %s "
                "FROM pg_catalog.pg_class c "
                "WHERE c.oid = 'action_authorization_claim.nonce_claims'::regclass",
                (_ROLE, _ROLE, _ROLE, _ROLE),
            )
            assert cursor.fetchone() == (True, False, False, True)
            cursor.execute(
                "SELECT routine_name FROM information_schema.routine_privileges "
                "WHERE routine_schema = 'action_authorization_claim' "
                "AND grantee = %s ORDER BY routine_name",
                (_ROLE,),
            )
            assert cursor.fetchall() == [("claim_action_authorization",)]

            cursor.execute(f"SET ROLE {_ROLE}")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cursor.execute(
                    "INSERT INTO action_authorization_claim.nonce_claims "
                    "(authorization_id, ticket_id) VALUES ('action-auth-12345678-1234-1234-1234-123456789abc', 'OMN-17462')"
                )
            connection.rollback()
    finally:
        connection.close()


@pytest.mark.integration
def test_postgres16_backout_revokes_interface_but_preserves_claim_history_table(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _provision_claim_schema(ephemeral_postgres)
    _apply(ephemeral_postgres, _MIGRATION)
    _apply(ephemeral_postgres, _ROLLBACK)
    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT "
                "to_regclass('action_authorization_claim.nonce_claims') IS NOT NULL, "
                "has_schema_privilege(%s, 'action_authorization_claim', 'USAGE')",
                (_ROLE,),
            )
            assert cursor.fetchone() == (True, False)
    finally:
        connection.close()
