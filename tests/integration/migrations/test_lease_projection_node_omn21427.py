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
from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    lint_application_database_sql,
)
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_FORWARD = Path(__file__).parents[3] / "docker/migrations/forward"
_MIGRATION = _FORWARD / "nodes/node_projection_lease/0000_create_lease_projection.sql"


def test_node_lease_migration_has_statically_provable_targets() -> None:
    assert (
        lint_application_database_sql(
            _MIGRATION.read_text(), load_topology_profile("local")
        )
        == ()
    )


@pytest.mark.parametrize(
    "null_column",
    [None, "kind", "resource", "lease_id", "fence", "holder", "state", "updated_at"],
)
def test_node_lease_migration_converges_without_replacing_existing_data(
    ephemeral_postgres: EphemeralPostgres, null_column: str | None
) -> None:
    pg = ephemeral_postgres
    setup = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE ROLE tenant_projection_writer; "
        "CREATE TABLE public.lease_projection (kind TEXT, resource TEXT, "
        "lease_id UUID, fence BIGINT, holder TEXT, state TEXT, updated_at TIMESTAMPTZ); "
        "INSERT INTO public.lease_projection VALUES ('slot', 'slot:h201', "
        "'00000000-0000-0000-0000-000000000007', 7, 'holder', 'held', '2026-10-27Z');",
    )
    assert setup.returncode == 0, setup.stderr
    if null_column is not None:
        null_updates = {
            "kind": "UPDATE public.lease_projection SET kind = NULL",
            "resource": "UPDATE public.lease_projection SET resource = NULL",
            "lease_id": "UPDATE public.lease_projection SET lease_id = NULL",
            "fence": "UPDATE public.lease_projection SET fence = NULL",
            "holder": "UPDATE public.lease_projection SET holder = NULL",
            "state": "UPDATE public.lease_projection SET state = NULL",
            "updated_at": "UPDATE public.lease_projection SET updated_at = NULL",
        }
        nulled = pg.psql("-c", null_updates[null_column])
        assert nulled.returncode == 0, nulled.stderr
    before = pg.psql("-Atc", "SELECT row_to_json(p) FROM public.lease_projection p")
    assert before.returncode == 0, before.stderr
    applied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_MIGRATION))
    if null_column is None:
        assert applied.returncode == 0, applied.stderr
        reapplied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(_MIGRATION))
        assert reapplied.returncode == 0, reapplied.stderr
        nullable = pg.psql(
            "-Atc",
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'lease_projection' "
            "AND is_nullable = 'YES'",
        )
        assert nullable.returncode == 0, nullable.stderr
        assert nullable.stdout.strip() == "0"
    else:
        assert applied.returncode != 0
        assert (
            f'column "{null_column}" of relation "lease_projection" contains null values'
            in applied.stderr
        )
    after = pg.psql("-Atc", "SELECT row_to_json(p) FROM public.lease_projection p")
    assert after.returncode == 0, after.stderr
    assert after.stdout == before.stdout


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
