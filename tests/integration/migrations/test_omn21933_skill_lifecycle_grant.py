# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21933 live apply of the skill lifecycle migrations on a disposable cluster."""

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.ephemeral_pg]

_FORWARD = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"
_ROLE = _FORWARD / "103_create_tenant_projection_writer_role.sql"
_VENDOR = _FORWARD / "nodes" / "node_projection_skill_lifecycle"


def _apply(pg: EphemeralPostgres, path: Path) -> None:
    result = pg.psql("-X", "-v", "ON_ERROR_STOP=1", "-f", str(path))
    assert result.returncode == 0, result.stderr


def _seed_populated_incomplete_table(pg: EphemeralPostgres) -> None:
    seed = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        """
        CREATE TABLE public.skill_executions (
            event_id UUID, run_id UUID, event_type TEXT, skill_name TEXT,
            repo_id TEXT, correlation_id UUID, emitted_at TIMESTAMPTZ, status TEXT
        );
        INSERT INTO public.skill_executions
            (event_id, run_id, event_type, skill_name, repo_id,
             correlation_id, emitted_at, status)
        VALUES ('00000000-0000-0000-0000-000000000011',
                '00000000-0000-0000-0000-000000000012', 'completed',
                'legacy-row', 'omnibase_infra',
                '00000000-0000-0000-0000-000000000013',
                '2026-01-01T00:00:00Z', 'success');
        """,
    )
    assert seed.returncode == 0, seed.stderr


@pytest.mark.parametrize(
    "initial_shape",
    ["fresh", "missing-columns", "populated-missing-columns", "legacy-flat"],
)
def test_writer_holds_select_insert_update_after_the_vendored_migrations(
    ephemeral_postgres: EphemeralPostgres,
    initial_shape: str,
) -> None:
    if initial_shape == "missing-columns":
        seed = ephemeral_postgres.psql(
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "CREATE TABLE public.skill_executions (event_id UUID)",
        )
        assert seed.returncode == 0, seed.stderr
    elif initial_shape == "populated-missing-columns":
        _seed_populated_incomplete_table(ephemeral_postgres)
    elif initial_shape == "legacy-flat":
        _apply(ephemeral_postgres, _FORWARD / "048_create_skill_executions.sql")
        seed = ephemeral_postgres.psql(
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            """
            INSERT INTO public.skill_executions
                (event_id, run_id, event_type, skill_name, repo_id,
                 correlation_id, emitted_at)
            VALUES ('00000000-0000-0000-0000-000000000011',
                    gen_random_uuid(), 'started', 'legacy-row',
                    'omnibase_infra', gen_random_uuid(), NOW())
            """,
        )
        assert seed.returncode == 0, seed.stderr
    _apply(ephemeral_postgres, _ROLE)
    _apply(ephemeral_postgres, _VENDOR / "0001_create_skill_executions.sql")
    _apply(
        ephemeral_postgres,
        _VENDOR / "0002_grant_tenant_projection_writer_skill_executions.sql",
    )
    for privilege, expected in (
        ("SELECT", "t"),
        ("INSERT", "t"),
        ("UPDATE", "t"),
        ("DELETE", "f"),
    ):
        probe = ephemeral_postgres.psql(
            "-At",
            "-c",
            "SELECT has_table_privilege("
            f"'tenant_projection_writer', 'public.skill_executions', '{privilege}')",
        )
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == expected, privilege

    unrelated = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-At",
        "-c",
        """
        CREATE ROLE lifecycle_unrelated NOLOGIN;
        GRANT USAGE ON SCHEMA public TO lifecycle_unrelated;
        SELECT has_table_privilege(
            'lifecycle_unrelated', 'public.skill_executions', 'SELECT')
            OR has_table_privilege(
                'lifecycle_unrelated', 'public.skill_executions', 'INSERT')
            OR has_table_privilege(
                'lifecycle_unrelated', 'public.skill_executions', 'UPDATE');
        """,
    )
    assert unrelated.returncode == 0, unrelated.stderr
    assert "f" in unrelated.stdout.splitlines(), unrelated.stdout
    unrelated_read = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "SET ROLE lifecycle_unrelated; SELECT * FROM public.skill_executions",
    )
    assert unrelated_read.returncode != 0
    assert "permission denied" in unrelated_read.stderr

    # Exercise the declared principal, including the projection's conflict path.
    write = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        """
        SET ROLE tenant_projection_writer;
        SET app.tenant_id = 'omninode';
        INSERT INTO public.skill_executions
            (event_id, run_id, event_type, skill_name, repo_id,
             correlation_id, emitted_at)
        VALUES ('00000000-0000-0000-0000-000000000001',
                '00000000-0000-0000-0000-000000000002', 'started',
                'landing-proof', 'omnibase_infra',
                '00000000-0000-0000-0000-000000000003', NOW())
        ON CONFLICT (event_id) DO UPDATE SET skill_name = EXCLUDED.skill_name;
        UPDATE public.skill_executions SET skill_name = 'readback'
            WHERE event_id = '00000000-0000-0000-0000-000000000001';
        SELECT skill_name FROM public.skill_executions;
        """,
    )
    assert write.returncode == 0, write.stderr
    assert "readback" in write.stdout
    _apply(ephemeral_postgres, _VENDOR / "0001_create_skill_executions.sql")
    _apply(
        ephemeral_postgres,
        _VENDOR / "0002_grant_tenant_projection_writer_skill_executions.sql",
    )
    readback = ephemeral_postgres.psql(
        "-At",
        "-c",
        "SELECT count(*) FROM public.skill_executions WHERE skill_name = 'readback'",
    )
    assert readback.returncode == 0, readback.stderr
    assert readback.stdout.strip() == "1"
    for tenant in ("omninode", "another-tenant", "", None):
        # One psql process is one session, so SET ROLE and the tenant binding
        # carry into the count. The tenant values are fixed literals of this
        # test, and the count is read from a labelled line rather than the
        # last line of whatever the session printed.
        tenant_args = (
            ("-c", f"SET app.tenant_id = '{tenant}'") if tenant is not None else ()
        )
        visible = ephemeral_postgres.psql(
            "-At",
            "-c",
            "SET ROLE tenant_projection_writer",
            *tenant_args,
            "-c",
            "SELECT 'bound=' || coalesce(current_setting('app.tenant_id', true), 'NULL')"
            " || ' visible=' || count(*) FROM public.skill_executions"
            " WHERE skill_name = 'readback'",
        )
        assert visible.returncode == 0, visible.stderr
        assert (
            f"bound={'NULL' if tenant is None else tenant} visible={1 if tenant == 'omninode' else 0}"
            in visible.stdout.splitlines()
        ), visible.stdout
    for tenant_setting in (
        (),
        ("-c", "SET app.tenant_id = 'another-tenant'"),
        ("-c", "SET app.tenant_id = ''"),
    ):
        denied_write = ephemeral_postgres.psql(
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "SET ROLE tenant_projection_writer",
            *tenant_setting,
            "-c",
            """
            INSERT INTO public.skill_executions
                (event_id, run_id, event_type, skill_name, repo_id,
                 correlation_id, emitted_at)
            VALUES (gen_random_uuid(), gen_random_uuid(), 'started',
                    'denied', 'omnibase_infra', gen_random_uuid(), NOW())
            """,
        )
        assert denied_write.returncode != 0
        assert "row-level security policy" in denied_write.stderr
    posture = ephemeral_postgres.psql(
        "-At",
        "-c",
        """
        SELECT relrowsecurity AND relforcerowsecurity
        FROM pg_catalog.pg_class WHERE oid = 'public.skill_executions'::regclass
        """,
    )
    assert posture.returncode == 0, posture.stderr
    assert posture.stdout.strip() == "t"
    if initial_shape in {"legacy-flat", "populated-missing-columns"}:
        legacy = ephemeral_postgres.psql(
            "-At",
            "-c",
            """
            SELECT tenant_id FROM public.skill_executions
            WHERE event_id = '00000000-0000-0000-0000-000000000011'
              AND skill_name = 'legacy-row'
            """,
        )
        assert legacy.returncode == 0, legacy.stderr
        assert legacy.stdout.strip() == "omninode"
        legacy_visible = ephemeral_postgres.psql(
            "-At",
            "-c",
            "SET ROLE tenant_projection_writer",
            "-c",
            "SET app.tenant_id = 'omninode'",
            "-c",
            "SELECT 'legacy=' || count(*) FROM public.skill_executions"
            " WHERE skill_name = 'legacy-row'",
        )
        assert legacy_visible.returncode == 0, legacy_visible.stderr
        assert "legacy=1" in legacy_visible.stdout.splitlines(), legacy_visible.stdout
    if initial_shape == "populated-missing-columns":
        reconciled = ephemeral_postgres.psql(
            "-At",
            "-c",
            """
            SELECT run_id = '00000000-0000-0000-0000-000000000012'::uuid
                AND event_type = 'completed' AND status = 'success'
                AND repo_id = 'omnibase_infra'
                AND correlation_id = '00000000-0000-0000-0000-000000000013'::uuid
                AND emitted_at = '2026-01-01T00:00:00Z'::timestamptz
                AND received_at IS NOT NULL AND started_emit_failed = FALSE
                AND skill_id IS NULL AND args_count IS NULL
                AND duration_ms IS NULL AND error_type IS NULL AND session_id IS NULL
            FROM public.skill_executions WHERE skill_name = 'legacy-row'
            """,
            "-c",
            """
            SELECT string_agg(attname, ',' ORDER BY attname)
            FROM pg_catalog.pg_attribute
            WHERE attrelid = 'public.skill_executions'::regclass
                AND attnum > 0 AND NOT attisdropped AND attnotnull;
            """,
        )
        assert reconciled.returncode == 0, reconciled.stderr
        assert reconciled.stdout.splitlines() == [
            "t",
            "correlation_id,emitted_at,event_id,event_type,received_at,repo_id,"
            "run_id,skill_name,tenant_id",
        ]
    denied = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "SET ROLE tenant_projection_writer; DELETE FROM public.skill_executions",
    )
    assert denied.returncode != 0
    assert "permission denied" in denied.stderr


@pytest.mark.parametrize(
    ("mutation_sql", "constraint", "row_count"),
    [
        (
            "UPDATE public.skill_executions SET event_type = 'invalid'",
            "skill_executions_event_type_check",
            "1",
        ),
        (
            "UPDATE public.skill_executions SET status = 'invalid'",
            "skill_executions_status_check",
            "1",
        ),
        (
            "INSERT INTO public.skill_executions SELECT * FROM public.skill_executions",
            "skill_executions_pkey",
            "2",
        ),
    ],
    ids=["invalid-event-type", "invalid-status", "duplicate-event-id"],
)
def test_populated_reconciliation_refuses_invalid_constraints_without_losing_rows(
    ephemeral_postgres: EphemeralPostgres,
    mutation_sql: str,
    constraint: str,
    row_count: str,
) -> None:
    _seed_populated_incomplete_table(ephemeral_postgres)
    mutation = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-c", mutation_sql)
    assert mutation.returncode == 0, mutation.stderr
    result = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(_VENDOR / "0001_create_skill_executions.sql"),
    )
    assert result.returncode != 0
    assert constraint in result.stderr
    readback = ephemeral_postgres.psql(
        "-At",
        "-c",
        "SELECT count(*) FROM public.skill_executions"
        " WHERE event_id = '00000000-0000-0000-0000-000000000011'"
        " AND skill_name = 'legacy-row'",
    )
    assert readback.returncode == 0, readback.stderr
    assert readback.stdout.strip() == row_count


def test_deploy_chain_targets_public_with_a_different_search_path(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    result = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA shadow",
        "-c",
        "SET search_path TO shadow, public",
        "-f",
        str(_ROLE),
        "-f",
        str(_VENDOR / "0001_create_skill_executions.sql"),
        "-f",
        str(_VENDOR / "0002_grant_tenant_projection_writer_skill_executions.sql"),
        "-At",
        "-c",
        """
        SELECT 'chain=' || (
            to_regclass('public.skill_executions') IS NOT NULL
            AND to_regclass('shadow.skill_executions') IS NULL
            AND EXISTS (
                SELECT 1 FROM pg_catalog.pg_policies
                WHERE schemaname = 'public' AND tablename = 'skill_executions'
                  AND policyname = 'tenant_isolation')
            AND has_table_privilege(
                'tenant_projection_writer', 'public.skill_executions', 'SELECT')
            AND has_table_privilege(
                'tenant_projection_writer', 'public.skill_executions', 'INSERT')
            AND has_table_privilege(
                'tenant_projection_writer', 'public.skill_executions', 'UPDATE')
        )
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "chain=true" in result.stdout.splitlines(), result.stdout


def test_incomplete_populated_table_refuses_to_invent_required_values(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    seed = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        """
        CREATE TABLE public.skill_executions (event_id UUID);
        INSERT INTO public.skill_executions
            VALUES ('00000000-0000-0000-0000-000000000001');
        """,
    )
    assert seed.returncode == 0, seed.stderr
    result = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(_VENDOR / "0001_create_skill_executions.sql"),
    )
    assert result.returncode != 0
    assert 'column "run_id"' in result.stderr
    assert "contains null values" in result.stderr
    readback = ephemeral_postgres.psql(
        "-At",
        "-c",
        "SELECT event_id FROM public.skill_executions",
    )
    assert readback.returncode == 0, readback.stderr
    assert readback.stdout.strip() == "00000000-0000-0000-0000-000000000001"
