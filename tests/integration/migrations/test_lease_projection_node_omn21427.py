# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The tenant projection writer can apply lease intents in its own database."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import asyncpg
import pytest

from omnibase_infra.handlers.lease import HandlerLeaseProjection, reduce_lease_event
from omnibase_infra.models.lease import ModelLeaseReply
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_FORWARD = Path(__file__).parents[3] / "docker/migrations/forward"
_MIGRATION = _FORWARD / "nodes/node_projection_lease/0000_create_lease_projection.sql"


async def test_application_database_lease_projection_applies_fenced_intents(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    created = pg.psql(
        "-v", "ON_ERROR_STOP=1", "-c", "CREATE DATABASE omnidash_analytics"
    )
    assert created.returncode == 0, created.stderr
    role = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(_FORWARD / "103_create_tenant_projection_writer_role.sql"),
    )
    assert role.returncode == 0, role.stderr
    for _ in range(2):
        applied = pg.psql(
            "-v", "ON_ERROR_STOP=1", "-f", str(_MIGRATION), dbname="omnidash_analytics"
        )
        assert applied.returncode == 0, applied.stderr
    login = pg.psql("-c", "ALTER ROLE tenant_projection_writer LOGIN")
    assert login.returncode == 0, login.stderr
    writer = await asyncpg.create_pool(
        host=pg.socket_dir,
        port=pg.port,
        user="tenant_projection_writer",
        database="omnidash_analytics",
        min_size=1,
        max_size=1,
    )
    try:
        projection = HandlerLeaseProjection(writer)
        for fence, holder in ((7, "first"), (9, "new"), (8, "stale")):
            _, intents = reduce_lease_event(
                {},
                ModelLeaseReply(
                    outcome="lease-granted",
                    kind="slot",
                    resource="slot:h201",
                    holder=holder,
                    lease_id=UUID(int=fence),
                    fence=fence,
                ),
            )
            assert len(intents) == 1
            await projection.apply(intents[0])
        async with writer.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT fence, holder, state FROM public.lease_projection"
            )
            assert row is not None
            assert tuple(row) == (9, "new", "held")
            assert not await connection.fetchval(
                "SELECT has_table_privilege(current_user, 'public.lease_projection', 'DELETE')"
            )
    finally:
        await writer.close()


def test_application_database_lease_migration_refuses_a_missing_write_grant(
    ephemeral_postgres: EphemeralPostgres, tmp_path: Path
) -> None:
    pg = ephemeral_postgres
    role = pg.psql("-c", "CREATE ROLE tenant_projection_writer")
    assert role.returncode == 0, role.stderr
    mutated = tmp_path / "partial_lease_grant.sql"
    mutated.write_text(
        _MIGRATION.read_text().replace(
            "GRANT SELECT, INSERT, UPDATE ON", "GRANT SELECT, INSERT ON", 1
        )
    )
    applied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(mutated))
    assert applied.returncode != 0
    assert "missing required tenant_projection_writer privilege" in applied.stderr
