# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live migration proof for hook-event envelope identity (OMN-17201).

The test uses only the throwaway Unix-socket Postgres cluster from this test
package.  It applies the real vendored SQL through ``psql -v ON_ERROR_STOP=1
-f`` in deployment order, seeds a legacy row before 0003, and inspects the
resulting catalog and data.  The connection is deliberately the fixture's
superuser: this proves DDL and data preservation, not RLS authorization.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import psycopg2.errors
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_hook_event_capture"
)
MIGRATION_0001 = MIGRATIONS_DIR / "0001_create_hook_events.sql"
MIGRATION_0002 = MIGRATIONS_DIR / "0002_hook_events_tenant_rls.sql"
MIGRATION_0003 = MIGRATIONS_DIR / "0003_add_hook_events_envelope_id.sql"
MANIFEST = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)
VERSION = "node:node_hook_event_capture:0003_add_hook_events_envelope_id.sql"
CHECKSUM = "03e471f301670f158f8a429dcd27f8d3f6da2c8fe76257ccbe1b3919e79d054a"
LEGACY_TENANT = "legacy-tenant"
LEGACY_EVENT_SHA = "a" * 64
CURRENT_TENANT = "current-tenant"
CURRENT_EVENT_SHA = "b" * 64
CURRENT_EVENT_ID = "producer-correlation-42"
ENVELOPE_ID = "11111111-2222-3333-4444-555555555555"
REDELIVERY_ENVELOPE_ID = "99999999-8888-7777-6666-555555555555"


def _apply(pg: EphemeralPostgres, migration: Path) -> subprocess.CompletedProcess[str]:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
    assert result.returncode == 0, (
        f"{migration.name} failed through psql -v ON_ERROR_STOP=1 -f:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _create_constrained_dashboard_role(pg: EphemeralPostgres) -> None:
    """Provide the role 0002 requires before it grants its RLS read surface."""
    connection = pg.connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE ROLE app_dashboard NOLOGIN NOSUPERUSER NOBYPASSRLS "
                "NOCREATEDB NOCREATEROLE NOREPLICATION"
            )
    finally:
        connection.close()


def _column_shape(pg: EphemeralPostgres) -> list[tuple[object, ...]]:
    connection = pg.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'hook_events' "
                "ORDER BY ordinal_position"
            )
            return list(cursor.fetchall())
    finally:
        connection.close()


def _insert_hook_event(
    pg: EphemeralPostgres,
    *,
    tenant_id: str,
    event_sha: str,
    event_id: str | None,
    envelope_id: str | None = None,
) -> None:
    connection = pg.connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if envelope_id is None:
                cursor.execute(
                    "INSERT INTO public.hook_events "
                    "(tenant_id, event_sha, event_type, occurred_at, payload, event_id, "
                    "source, batch_sha) "
                    "VALUES (%s, %s, 'hook.event', now(), '{}'::jsonb, %s, "
                    "'gateway', %s)",
                    (tenant_id, event_sha, event_id, "c" * 64),
                )
            else:
                cursor.execute(
                    "INSERT INTO public.hook_events "
                    "(tenant_id, event_sha, event_type, occurred_at, payload, event_id, "
                    "source, batch_sha, envelope_id) "
                    "VALUES (%s, %s, 'hook.event', now(), '{}'::jsonb, %s, "
                    "'gateway', %s, %s)",
                    (tenant_id, event_sha, event_id, "c" * 64, envelope_id),
                )
    finally:
        connection.close()


def _hook_event_rows(
    pg: EphemeralPostgres,
) -> list[tuple[object, ...]]:
    connection = pg.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT tenant_id, event_sha::text, event_id, envelope_id::text "
                "FROM public.hook_events ORDER BY tenant_id, event_sha"
            )
            return list(cursor.fetchall())
    finally:
        connection.close()


def _manifest_checksum(version: str) -> str:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 6 and fields[4] == version:
            return fields[5]
    raise AssertionError(f"{version} is missing from {MANIFEST}")


def test_0003_matches_its_declared_manifest_checksum() -> None:
    """The vendored migration is bound to its checked-in apply declaration."""
    vendored_bytes = MIGRATION_0003.read_bytes()
    assert hashlib.sha256(vendored_bytes).hexdigest() == CHECKSUM
    assert _manifest_checksum(VERSION) == CHECKSUM


def test_0003_preserves_legacy_content_identity_and_adds_nullable_envelope_id(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """0003 adds only a nullable UUID trace; content dedupe remains unchanged."""
    _apply(ephemeral_postgres, MIGRATION_0001)
    _create_constrained_dashboard_role(ephemeral_postgres)
    _apply(ephemeral_postgres, MIGRATION_0002)

    before_0003 = _column_shape(ephemeral_postgres)
    _insert_hook_event(
        ephemeral_postgres,
        tenant_id=LEGACY_TENANT,
        event_sha=LEGACY_EVENT_SHA,
        event_id=None,
    )

    _apply(ephemeral_postgres, MIGRATION_0003)
    assert _column_shape(ephemeral_postgres) == [
        *before_0003,
        ("envelope_id", "uuid", "YES", None),
    ]

    _insert_hook_event(
        ephemeral_postgres,
        tenant_id=CURRENT_TENANT,
        event_sha=CURRENT_EVENT_SHA,
        event_id=CURRENT_EVENT_ID,
        envelope_id=ENVELOPE_ID,
    )
    # The tenant is part of content identity. The same content address is valid
    # for another tenant, and 0003 intentionally gives envelope_id no unique key.
    _insert_hook_event(
        ephemeral_postgres,
        tenant_id=LEGACY_TENANT,
        event_sha=CURRENT_EVENT_SHA,
        event_id=CURRENT_EVENT_ID,
        envelope_id=ENVELOPE_ID,
    )

    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert_hook_event(
            ephemeral_postgres,
            tenant_id=CURRENT_TENANT,
            event_sha=CURRENT_EVENT_SHA,
            event_id=CURRENT_EVENT_ID,
            envelope_id=REDELIVERY_ENVELOPE_ID,
        )

    assert _hook_event_rows(ephemeral_postgres) == [
        (CURRENT_TENANT, CURRENT_EVENT_SHA, CURRENT_EVENT_ID, ENVELOPE_ID),
        (LEGACY_TENANT, LEGACY_EVENT_SHA, None, None),
        (LEGACY_TENANT, CURRENT_EVENT_SHA, CURRENT_EVENT_ID, ENVELOPE_ID),
    ]

    # The real SQL promises idempotence with ADD COLUMN IF NOT EXISTS. Reapply
    # after data exists so the claim includes preservation, not merely DDL parse.
    _apply(ephemeral_postgres, MIGRATION_0003)
    assert _hook_event_rows(ephemeral_postgres) == [
        (CURRENT_TENANT, CURRENT_EVENT_SHA, CURRENT_EVENT_ID, ENVELOPE_ID),
        (LEGACY_TENANT, LEGACY_EVENT_SHA, None, None),
        (LEGACY_TENANT, CURRENT_EVENT_SHA, CURRENT_EVENT_ID, ENVELOPE_ID),
    ]
