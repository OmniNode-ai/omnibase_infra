# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply the vendored automation-liveness migrations on real PostgreSQL 16."""

from __future__ import annotations

from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_ROOT = Path(__file__).resolve().parents[3]
_MIGRATIONS = (
    _ROOT / "docker/migrations/forward/nodes/node_projection_automation_liveness"
)
_FILES = (
    "0000_create_automation_liveness.sql",
    "0001_grant_omninode_runtime_automation_liveness.sql",
)
_TABLES = (
    "automation_liveness_state",
    "automation_run_history",
    "automation_alarm_episodes",
)


def _apply(pg: EphemeralPostgres) -> None:
    for filename in _FILES:
        migration = _MIGRATIONS / filename
        assert migration.is_file(), f"missing vendored migration: {migration}"
        applied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
        assert applied.returncode == 0, applied.stdout + applied.stderr


def test_migrations_deliver_runtime_writes_and_converge(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The real apply path delivers usable ACLs and preserves persisted rows.

    Revoke one required grant as a positive control: the same runtime upsert
    must then fail, and replaying the grant migration must restore it.
    """
    pg = ephemeral_postgres
    provisioned = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA omninode_internal; "
        "CREATE ROLE omninode_runtime NOLOGIN NOSUPERUSER NOBYPASSRLS; "
        "CREATE ROLE automation_liveness_test LOGIN NOINHERIT "
        "NOSUPERUSER NOBYPASSRLS; "
        "GRANT omninode_runtime TO automation_liveness_test;",
    )
    assert provisioned.returncode == 0, provisioned.stderr
    _apply(pg)

    upsert = """
        INSERT INTO omninode_internal.automation_liveness_state
            (process_key, process_id, host, process_state, projected_at)
        VALUES ('watchdog@lab', 'watchdog', 'lab', 'declared', NOW())
        ON CONFLICT (process_key) DO UPDATE SET process_state = 'running'
    """
    with pg.connect(user="automation_liveness_test") as runtime:
        with runtime.cursor() as cur:
            cur.execute("SET ROLE omninode_runtime")
            cur.execute(
                "SELECT current_user, rolsuper, rolbypassrls "
                "FROM pg_roles WHERE rolname = current_user"
            )
            assert cur.fetchone() == ("omninode_runtime", False, False)
            cur.execute(upsert)
            cur.execute(upsert)
            cur.execute("""
                INSERT INTO omninode_internal.automation_run_history
                    (run_key, process_id, host, run_id, phase, started_at,
                     evidence_ref, emitter, observed_at, contract_digest,
                     projected_at)
                VALUES ('watchdog@lab#run#finished', 'watchdog', 'lab', 'run',
                        'finished', NOW(), 'test-evidence', 'test-emitter', NOW(),
                        'test-digest', NOW())
            """)
            cur.execute("""
                INSERT INTO omninode_internal.automation_alarm_episodes
                    (episode_id, projected_at)
                VALUES ('test-episode', NOW())
            """)
            cur.execute("""
                UPDATE omninode_internal.automation_run_history
                SET outcome = 'passed' WHERE run_id = 'run'
            """)
            assert cur.rowcount == 1
            cur.execute("""
                UPDATE omninode_internal.automation_alarm_episodes
                SET state = 'cleared' WHERE episode_id = 'test-episode'
            """)
            assert cur.rowcount == 1
            before = {}
            for table in _TABLES:
                cur.execute(
                    sql.SQL("SELECT * FROM omninode_internal.{}").format(
                        sql.Identifier(table)
                    )
                )
                before[table] = cur.fetchall()
                assert len(before[table]) == 1
                cur.execute(
                    "SELECT has_table_privilege(current_user, %s, 'DELETE')",
                    (f"omninode_internal.{table}",),
                )
                assert cur.fetchone() == (False,)
        runtime.commit()

        _apply(pg)
        with runtime.cursor() as cur:
            for table in _TABLES:
                cur.execute(
                    sql.SQL("SELECT * FROM omninode_internal.{}").format(
                        sql.Identifier(table)
                    )
                )
                assert cur.fetchall() == before[table]
        runtime.commit()

        revoked = pg.psql(
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "REVOKE UPDATE ON omninode_internal.automation_liveness_state "
            "FROM omninode_runtime;",
        )
        assert revoked.returncode == 0, revoked.stderr
        with (
            runtime.cursor() as cur,
            pytest.raises(psycopg2.errors.InsufficientPrivilege),
        ):
            cur.execute(upsert)
        runtime.rollback()

        _apply(pg)
        with runtime.cursor() as cur:
            cur.execute(upsert)
            cur.execute(
                "SELECT process_state FROM omninode_internal.automation_liveness_state"
            )
            assert cur.fetchall() == [("running",)]
