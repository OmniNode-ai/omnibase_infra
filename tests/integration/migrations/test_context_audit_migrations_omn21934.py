# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply the context-audit migrations and exercise the writer's real upsert."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.ephemeral_pg]

_MIGRATIONS = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_context_audit"
)
_CREATE = _MIGRATIONS / "0000_add_context_audit_event_key.sql"
_GRANT = _MIGRATIONS / "0001_grant_tenant_projection_writer_context_audit_events.sql"
_ROLE = (
    "CREATE ROLE tenant_projection_writer NOLOGIN NOSUPERUSER NOBYPASSRLS "
    "NOCREATEDB NOCREATEROLE NOREPLICATION"
)


@pytest.mark.parametrize("shape", ["absent", "partial", "without_id"])
def test_migrations_reconcile_rerun_and_allow_idempotent_writer_upsert(
    ephemeral_postgres: EphemeralPostgres, shape: str
) -> None:
    pg = ephemeral_postgres
    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(_ROLE)
        if shape != "absent":
            id_column = "id BIGSERIAL PRIMARY KEY," if shape == "partial" else ""
            cursor.execute(
                f"CREATE TABLE public.context_audit_events ({id_column} "
                "task_id TEXT NOT NULL, correlation_id TEXT NOT NULL, "
                "event_type TEXT NOT NULL, enforcement_level TEXT NOT NULL)"
            )
            cursor.execute(
                "INSERT INTO public.context_audit_events "
                "(task_id, correlation_id, event_type, enforcement_level) "
                "VALUES ('legacy', 'correlation', 'audit', 'warn')"
            )

    for _ in range(2):
        for migration in (_CREATE, _GRANT):
            applied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
            assert applied.returncode == 0, applied.stderr

    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL ROLE tenant_projection_writer")
        ids = []
        for action in ("first", "replayed"):
            cursor.execute(
                "INSERT INTO public.context_audit_events "
                "(task_id, correlation_id, event_type, enforcement_level, "
                "event_key, enforcement_action) "
                "VALUES ('task', 'correlation', 'audit', 'warn', 'key', %s) "
                "ON CONFLICT (event_key) DO UPDATE "
                "SET enforcement_action = EXCLUDED.enforcement_action "
                "RETURNING id",
                (action,),
            )
            ids.append(cursor.fetchone()[0])
        assert ids[0] == ids[1]
        cursor.execute(
            "SELECT event_key, enforcement_action "
            "FROM public.context_audit_events ORDER BY id"
        )
        expected: list[tuple[str | None, str | None]] = [("key", "replayed")]
        if shape != "absent":
            expected.insert(0, (None, None))
        assert cursor.fetchall() == expected
        cursor.execute(
            "SELECT has_table_privilege(current_user, "
            "'public.context_audit_events', 'DELETE')"
        )
        assert cursor.fetchone() == (False,)


def test_missing_cluster_role_refuses_with_provisioning_hint(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    created = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_CREATE))
    assert created.returncode == 0, created.stderr
    refused = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_GRANT))
    assert refused.returncode != 0
    assert "provision the cluster role before node migrations" in refused.stderr
    assert "scripts/provision-cluster-roles.sh --apply" in refused.stderr
    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM pg_roles WHERE rolname = 'tenant_projection_writer'"
        )
        assert cursor.fetchone() == (0,)


def test_unexpected_sequence_refuses_before_any_table_grant(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    created = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_CREATE))
    assert created.returncode == 0, created.stderr
    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(_ROLE)
        cursor.execute(
            "ALTER SEQUENCE public.context_audit_events_id_seq "
            "RENAME TO restored_context_audit_id_seq"
        )
    refused = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_GRANT))
    assert refused.returncode != 0
    assert "id is not backed by context_audit_events_id_seq" in refused.stderr
    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT has_table_privilege('tenant_projection_writer', "
            "'public.context_audit_events', 'SELECT,INSERT,UPDATE')"
        )
        assert cursor.fetchone() == (False,)
