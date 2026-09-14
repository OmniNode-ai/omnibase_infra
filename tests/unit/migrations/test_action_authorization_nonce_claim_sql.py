# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline shape checks for the isolated durable nonce-claim migration."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION = (
    _ROOT / "docker/migrations/forward/107_create_action_authorization_nonce_claim.sql"
)
_ROLLBACK = (
    _ROOT
    / "docker/migrations/rollback/rollback_107_create_action_authorization_nonce_claim.sql"
)
_RUNNER = _ROOT / "scripts/run-forward-migrations.sh"


@pytest.mark.unit
def test_nonce_claim_migration_owns_only_its_dedicated_boundary() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")
    table_definition = sql.split("CREATE FUNCTION", maxsplit=1)[0]

    assert "FROM pg_catalog.pg_namespace" in sql
    assert "WHERE nspname = 'action_authorization_claim'" in sql
    assert "CREATE SCHEMA action_authorization_claim" not in sql
    assert "CREATE TABLE action_authorization_claim.nonce_claims" in sql
    assert "first_effect" not in sql
    assert "CASCADE" not in sql
    assert "nonce TEXT" not in table_definition
    for field in (
        "authorization_id",
        "ticket_id",
        "contract_path",
        "contract_commit_sha",
        "contract_sha256",
        "action_id",
        "source_sha",
        "artifact_sha256",
        "target_database",
        "target_schema",
        "target_service",
        "target_principal",
        "execute_enabled",
        "issuer",
        "nonce_digest",
        "issued_at",
        "expires_at",
        "one_time_use",
        "reason",
        "request_digest",
        "state",
        "version",
        "redacted_receipt_digest",
    ):
        assert field in table_definition


@pytest.mark.unit
def test_nonce_claim_migration_is_atomic_default_deny_and_acl_restricted() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")

    assert "CHECK (execute_enabled IS FALSE)" in sql
    assert "CHECK (one_time_use IS TRUE)" in sql
    assert "claim_time := clock_timestamp()" in sql
    assert "claim_time >= p_expires_at" in sql
    assert "SECURITY DEFINER" in sql

    # The claim is settled by one statement against the unique indexes. A
    # retry loop here is what produced the OMN-17486 livelock: the conflict
    # branch answered, fell through to the insert, caught its own unique
    # violation and went round again, so a second claim of the same request
    # spun forever with no wait event and no lock to point at. A loop that
    # cannot be seen to terminate by reading it does not belong in a SECURITY
    # DEFINER function every lane's migration runner applies.
    assert "ON CONFLICT DO NOTHING" in sql
    assert "LOOP" not in sql
    assert "unique_violation" not in sql
    assert "claim_action_authorization(" in sql
    assert "register_action_authorization" not in sql
    assert "NOT_FOUND" not in sql
    assert "REVOKED" not in sql
    assert (
        "REVOKE ALL ON TABLE action_authorization_claim.nonce_claims FROM PUBLIC" in sql
    )
    assert (
        "REVOKE ALL ON FUNCTION action_authorization_claim.claim_action_authorization"
        in sql
    )
    assert (
        "GRANT EXECUTE ON FUNCTION action_authorization_claim.claim_action_authorization"
        in sql
    )
    assert "GRANT ALL" not in sql
    assert "FROM rsd_action_authorization_claim" in sql
    assert "old first-effect" not in sql.lower()


@pytest.mark.unit
def test_backout_disables_restricted_claim_without_destroying_history() -> None:
    rollback = _ROLLBACK.read_text(encoding="utf-8")

    assert (
        "REVOKE EXECUTE ON FUNCTION action_authorization_claim.claim_action_authorization"
        in rollback
    )
    assert "REVOKE USAGE ON SCHEMA action_authorization_claim" in rollback
    assert "DROP " not in rollback
    assert "CASCADE" not in rollback


@pytest.mark.unit
def test_flat_migration_is_discovered_and_registered_by_service_ledger_runner() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")

    assert 'migration_id="docker/${filename}"' in runner
    assert (
        "INSERT INTO public.schema_migrations (migration_id, checksum, source_set)"
        in runner
    )
    assert _MIGRATION.is_file()
