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


def test_writer_holds_select_insert_update_after_the_vendored_migrations(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
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
