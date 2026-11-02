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


@pytest.mark.parametrize("initial_shape", ["fresh", "missing-columns", "legacy-flat"])
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
    for tenant in ("omninode", "another-tenant", None):
        tenant_args = ("-c", f"SET app.tenant_id = '{tenant}'") if tenant else ()
        visible = ephemeral_postgres.psql(
            "-At",
            "-c",
            "SET ROLE tenant_projection_writer",
            *tenant_args,
            "-c",
            "SELECT count(*) FROM public.skill_executions WHERE skill_name = 'readback'",
        )
        assert visible.returncode == 0, visible.stderr
        assert visible.stdout.strip().splitlines()[-1] == (
            "1" if tenant == "omninode" else "0"
        )
    for tenant_setting in ((), ("-c", "SET app.tenant_id = 'another-tenant'")):
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
    if initial_shape == "legacy-flat":
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
    denied = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "SET ROLE tenant_projection_writer; DELETE FROM public.skill_executions",
    )
    assert denied.returncode != 0
    assert "permission denied" in denied.stderr


def test_create_targets_public_with_a_different_search_path(
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
        str(_VENDOR / "0001_create_skill_executions.sql"),
        "-At",
        "-c",
        """
        SELECT to_regclass('public.skill_executions') IS NOT NULL
           AND to_regclass('shadow.skill_executions') IS NULL
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("t")


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
